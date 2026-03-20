/*
 * FCLS-enabled SRv6 P4 Switch
 *
 * Implements:
 *   - Ethernet / IPv6 / SRH header parsing
 *   - FCLS (Flexible Computing-aware Label Stack) TLV parsing
 *   - fcls_flow_table: fast-path cache for FCLS flows
 *   - srh_forward_table: standard SRv6 segment routing forwarding
 *   - Controller query trigger via digest/clone for unknown flows
 *   - Port status registers for state reporting
 *
 * Target: BMv2 simple_switch (v1model)
 */

#include <core.p4>
#include <v1model.p4>

/*========== Constants ==========*/

const bit<16> ETHERTYPE_IPV6 = 0x86DD;
const bit<8>  IP_PROTO_SRH   = 43;     // Routing Header
const bit<8>  SRH_ROUTING_TYPE = 4;    // SRv6

const bit<8>  FCLS_TLV_TASK_TYPE   = 0x01;
const bit<8>  FCLS_TLV_COMPUTE_REQ = 0x02;
const bit<8>  FCLS_TLV_LATENCY_REQ = 0x03;
const bit<8>  FCLS_TLV_DATA_SIZE   = 0x04;

typedef bit<9>  port_t;
typedef bit<48> mac_t;
typedef bit<128> ipv6_t;

/*========== Headers ==========*/

header ethernet_t {
    mac_t   dst_addr;
    mac_t   src_addr;
    bit<16> ether_type;
}

header ipv6_t {
    bit<4>   version;
    bit<8>   traffic_class;
    bit<20>  flow_label;
    bit<16>  payload_length;
    bit<8>   next_header;
    bit<8>   hop_limit;
    ipv6_t   src_addr;
    ipv6_t   dst_addr;
}

header srh_t {
    bit<8>  next_header;
    bit<8>  hdr_ext_len;
    bit<8>  routing_type;
    bit<8>  segments_left;
    bit<8>  last_entry;
    bit<8>  flags;
    bit<16> tag;
}

/* Simplified: first segment in the SID list */
header sid_t {
    ipv6_t  sid;
}

/* FCLS header - computing-aware label stack extension */
header fcls_t {
    bit<8>  version;
    bit<8>  task_type;       // Service type (inference=0x01, training=0x02, etc.)
    bit<16> compute_req;     // Required compute (normalized, TOPS/TFLOPS)
    bit<32> latency_req;     // Max latency constraint (ms)
    bit<32> data_size;       // Input data size (MB)
}

/* Metadata for internal processing */
struct metadata_t {
    bit<1>  has_fcls;
    bit<1>  cache_hit;
    bit<8>  task_type;
    bit<16> compute_req;
    bit<32> latency_req;
    bit<32> data_size;
    ipv6_t  original_dst;
}

struct headers_t {
    ethernet_t  ethernet;
    ipv6_t      ipv6;
    srh_t       srh;
    sid_t       sid0;         // Active SID
    fcls_t      fcls;
}

/*========== Parser ==========*/

parser FclsParser(
    packet_in pkt,
    out headers_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t std_meta
) {
    state start {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV6: parse_ipv6;
            default: accept;
        }
    }

    state parse_ipv6 {
        pkt.extract(hdr.ipv6);
        meta.original_dst = hdr.ipv6.dst_addr;
        transition select(hdr.ipv6.next_header) {
            IP_PROTO_SRH: parse_srh;
            default: accept;
        }
    }

    state parse_srh {
        pkt.extract(hdr.srh);
        transition select(hdr.srh.routing_type) {
            SRH_ROUTING_TYPE: parse_sid;
            default: accept;
        }
    }

    state parse_sid {
        pkt.extract(hdr.sid0);
        /* Check if FCLS header follows (indicated by SRH flags or TLV) */
        transition parse_fcls;
    }

    state parse_fcls {
        /* Try to extract FCLS header if present */
        pkt.extract(hdr.fcls);
        meta.has_fcls = 1;
        meta.task_type = hdr.fcls.task_type;
        meta.compute_req = hdr.fcls.compute_req;
        meta.latency_req = hdr.fcls.latency_req;
        meta.data_size = hdr.fcls.data_size;
        transition accept;
    }
}

/*========== Checksum Verification ==========*/

control FclsVerifyChecksum(
    inout headers_t hdr,
    inout metadata_t meta
) {
    apply { }
}

/*========== Ingress Processing ==========*/

control FclsIngress(
    inout headers_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t std_meta
) {
    /* --- Registers for port state monitoring --- */
    register<bit<32>>(512) port_queue_depth;
    register<bit<32>>(512) port_byte_count;

    /* --- Actions --- */

    action drop() {
        mark_to_drop(std_meta);
    }

    action set_srh_path(ipv6_t target_sid, port_t egress_port) {
        /*
         * Set the SRv6 path from controller response.
         * In practice, this would write a full SID list into the SRH.
         * Simplified: set the active SID and forward.
         */
        hdr.sid0.sid = target_sid;
        hdr.ipv6.dst_addr = target_sid;
        std_meta.egress_spec = egress_port;
        meta.cache_hit = 1;
    }

    action trigger_controller_query() {
        /*
         * Send packet metadata to controller via clone/digest.
         * The controller will compute the optimal path and install
         * flow entries. Original packet is sent to controller port.
         */
        clone3(CloneType.I2E, 100, { std_meta });
        meta.cache_hit = 0;
    }

    action ipv6_forward(port_t egress_port) {
        /*
         * Standard IPv6 forwarding based on destination address
         * (which is the current active SID).
         */
        std_meta.egress_spec = egress_port;
        hdr.ipv6.hop_limit = hdr.ipv6.hop_limit - 1;
    }

    action decrement_segments_left() {
        /*
         * SRv6 segment processing: decrement segments_left
         * and update destination to next SID.
         */
        hdr.srh.segments_left = hdr.srh.segments_left - 1;
    }

    /* --- Tables --- */

    /*
     * FCLS flow table: fast-path cache for computing-aware flows.
     * Match on src/dst IPv6 + task_type from FCLS header.
     * If hit: apply cached SRv6 path (set_srh_path).
     * If miss: trigger controller query.
     */
    table fcls_flow_table {
        key = {
            hdr.ipv6.src_addr : exact;
            hdr.ipv6.dst_addr : exact;
            meta.task_type    : exact;
        }
        actions = {
            set_srh_path;
            trigger_controller_query;
        }
        size = 1024;
        default_action = trigger_controller_query();
    }

    /*
     * SRH forwarding table: standard SRv6 forwarding.
     * Match on IPv6 destination (current active SID).
     */
    table srh_forward_table {
        key = {
            hdr.ipv6.dst_addr : exact;
        }
        actions = {
            ipv6_forward;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    /* --- Main ingress logic --- */

    apply {
        if (!hdr.ethernet.isValid()) {
            drop();
            return;
        }

        if (hdr.ipv6.isValid()) {
            /* Check hop limit */
            if (hdr.ipv6.hop_limit == 0) {
                drop();
                return;
            }

            /* If packet has FCLS header, try fast-path cache */
            if (meta.has_fcls == 1) {
                fcls_flow_table.apply();
            }

            /* SRv6 processing */
            if (hdr.srh.isValid() && hdr.srh.segments_left > 0) {
                decrement_segments_left();
            }

            /* Standard IPv6/SRv6 forwarding */
            srh_forward_table.apply();

            /* Update port statistics register */
            bit<32> cur_queue;
            port_queue_depth.read(cur_queue, (bit<32>)std_meta.egress_spec);
            port_queue_depth.write(
                (bit<32>)std_meta.egress_spec,
                cur_queue + 1
            );
        }
    }
}

/*========== Egress Processing ==========*/

control FclsEgress(
    inout headers_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t std_meta
) {
    apply { }
}

/*========== Checksum Computation ==========*/

control FclsComputeChecksum(
    inout headers_t hdr,
    inout metadata_t meta
) {
    apply { }
}

/*========== Deparser ==========*/

control FclsDeparser(
    packet_out pkt,
    in headers_t hdr
) {
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv6);
        pkt.emit(hdr.srh);
        pkt.emit(hdr.sid0);
        pkt.emit(hdr.fcls);
    }
}

/*========== Switch Pipeline ==========*/

V1Switch(
    FclsParser(),
    FclsVerifyChecksum(),
    FclsIngress(),
    FclsEgress(),
    FclsComputeChecksum(),
    FclsDeparser()
) main;
