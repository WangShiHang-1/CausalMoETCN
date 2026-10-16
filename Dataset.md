# CausalMoETCN: Core-Network Dataset

## 1. Dataset Overview

The core-network dataset was collected from a Kubernetes-based 5G core-network and radio access network (RAN) testbed built upon free5GC. The testbed consists of four base stations and 40 user equipment (UE) devices. Controlled faults and operational events were introduced to generate diverse network conditions and record the resulting network responses. The data were collected in seven independent batches (D01–D07) and aligned to a one-second time grid (1 Hz), although individual indicators may have different underlying update intervals.

Three throughput indicators are used as the main performance measures: N3 uplink bit rate, N3 downlink bit rate, and N6 aggregate bit rate. In addition, 18 network and system indicators are recorded, covering session activity, service performance, RAN status, and host resource utilization.

| Property | Description |
|---|---|
| Collection environment | Kubernetes-based free5GC 5G core-network and RAN testbed |
| Testbed scale | 4 base stations and 40 UE devices |
| Collection batches | 7 independent batches (D01–D07) |
| Temporal resolution | 1 s |
| Main performance indicators | 3 throughput KPIs |
| Additional indicators | 18 network and system indicators |
| Operational event types | 9 operation types |
| Service groups | Gold and Silver |
| Joint operation configurations | 12 |
| Planned event duration | 120 or 180 s |
| Historical window | 300 s |
| Retained windows | 1,006 |

## 2. Network and System Indicators

### 2.1 Main Performance Indicators

| Indicator | Source field | Unit |
|---|---|---|
| N3 uplink bit rate | `n3_uplink_bitrate_bps` | bit/s |
| N3 downlink bit rate | `n3_downlink_bitrate_bps` | bit/s |
| N6 aggregate bit rate | `n6_bitrate_bps` | bit/s |

N3 carries user-plane traffic between the RAN and the user plane function (UPF), while N6 connects the UPF to the external data network. These indicators characterize traffic changes across the core-network data path. Values are recorded in bit/s and converted to Mbps for visualization.

### 2.2 Additional Indicators

| Indicator | Source field | Unit / Representation |
|---|---|---|
| PDU session establishment success rate | `cp_pdu_session_success_rate` | % |
| PDU session establishment failure count | `cp_pdu_session_failure_count` | Count |
| End-to-end round-trip time | `end_to_end_rtt_ms` | ms |
| Delay jitter | `jitter_ms` | ms |
| Packet-loss rate | `packet_loss_rate` | Ratio |
| Business completion rate | `business_completion_rate` | Ratio |
| Active PDU session count | `cp_active_session_count` | Count |
| Pending PDU session count | `cp_pending_session_count` | Count |
| PDU session request count | `cp_pdu_session_request_count` | Count |
| UE registration success rate | `registration_success_rate` | % |
| UE registration request count | `registration_request_count` | Count |
| RAN active PDU count | `ran_pdu_active_count` | Count |
| Ready base-station count | `ready_gnb_count` | Count |
| Observed UE count | `observed_ue_count` | Count |
| N4 bit rate | `n4_bitrate_bps` | bit/s |
| Host CPU utilization | `host_cpu_pct` | % |
| Host memory utilization | `host_mem_used_pct` | % |
| Host one-minute load average | `host_loadavg_1` | Dimensionless |

The indicators cover four main aspects of the testbed: session and registration activity, service-level performance, RAN status, and host resource utilization. Count-based indicators are retained in their recorded form, while snapshot indicators describe the network state at each observation time.

## 3. Operational Events

The dataset contains nine operation types, including six core-network faults and three RAN operations. Events can be applied to either the Gold or Silver service group.

| Category | Operation | Encoding field |
|---|---|---|
| Core network | N3 blocking | `core_n3_block` |
| Core network | N3 delay injection | `core_n3_delay` |
| Core network | N3 packet-loss injection | `core_n3_loss` |
| Core network | N4 blocking | `core_n4_block` |
| Core network | N6 blocking | `core_n6_block` |
| Core network | UPF shutdown | `core_upf_stop` |
| RAN | Base-station restart | `ran_gnb_restart` |
| RAN | UE recreation | `ran_ue_recreate` |
| RAN | UE scale-down | `ran_ue_scale_down` |

The observed data contain 12 joint operation configurations, formed by combining one core-network fault, one RAN operation, and one service group. The configurations are:

| Core-network fault | Gold group | Silver group |
|---|---|---|
| N3 blocking | Base-station restart | UE scale-down |
| N3 delay injection | UE scale-down | UE recreation |
| N3 packet-loss injection | UE recreation | Base-station restart |
| N4 blocking | Base-station restart | UE scale-down |
| N6 blocking | UE scale-down | UE recreation |
| UPF shutdown | UE recreation | Base-station restart |

A no-operation condition is also included as the baseline condition. Operational events were scheduled for either 120 or 180 s. The recorded network responses may persist beyond the scheduled event duration because service recovery is not necessarily instantaneous.

## 4. Data Windowing and Split

The retained dataset contains 503 intervention windows and 503 matched non-intervention windows, resulting in 1,006 windows in total. Each window contains 300 s of historical observations and is associated with an operational event or a corresponding non-intervention condition.

The data are divided by collection batch rather than by randomly shuffling individual windows, reducing potential information leakage between training and evaluation sets.

| Split | Collection batches | Intervention windows | Control windows | Total |
|---|---|---:|---:|---:|
| Training | D01–D04 | 288 | 288 | 576 |
| Validation | D05 | 72 | 72 | 144 |
| Test | D06–D07 | 143 | 143 | 286 |
| **Total** | **D01–D07** | **503** | **503** | **1,006** |

## 5. Data Preprocessing

The telemetry records are aligned to a one-second time grid. Historical missing observations are filled using available past observations, with training-set statistics used as fallback values when necessary. Missing target measurements are excluded from subsequent evaluation. The three throughput indicators are transformed using a logarithmic transformation relative to a historical baseline and then standardized using statistics calculated from the training set only. The same transformation parameters are applied to the validation and test sets.
