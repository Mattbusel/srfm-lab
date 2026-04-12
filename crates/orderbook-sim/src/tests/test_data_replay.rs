//! test_data_replay.rs — Integration tests for data_replay module.
//! Tests FIX parsing, ITCH parsing, PCAP handling, book reconstruction.

use crate::data_replay::*;

// ── FIX parser tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod fix_tests {
    use super::*;

    fn make_fix_logon() -> Vec<u8> {
        b"8=FIX.4.2\x019=49\x0135=A\x0149=SENDER\x0156=TARGET\x0134=1\x0152=20240101-09:00:00\x0198=0\x01108=30\x0110=100\x01".to_vec()
    }

    fn make_fix_new_order() -> Vec<u8> {
        b"8=FIX.4.2\x019=80\x0135=D\x0149=CLIENT\x0156=EXCHANGE\x0134=5\x0152=20240101-09:30:00\x0111=ORDER001\x0155=AAPL\x0154=1\x0138=100\x0140=2\x0144=150.25\x0110=099\x01".to_vec()
    }

    fn make_fix_exec_report() -> Vec<u8> {
        b"8=FIX.4.2\x019=100\x0135=8\x0149=EXCHANGE\x0156=CLIENT\x0134=10\x0152=20240101-09:30:01\x0111=ORDER001\x0137=EX001\x0117=EXEC001\x0120=0\x01150=2\x0139=2\x0154=1\x0138=100\x0132=100\x0131=150.25\x0114=100\x06151=0\x016=150.25\x0110=155\x01".to_vec()
    }

    #[test]
    fn test_parse_logon() {
        let mut parser = FixParser::new(FixVersion::Fix42);
        let raw = make_fix_logon();
        let result = parser.parse_message(&raw);
        assert!(result.is_ok(), "parse failed: {:?}", result);
        let (msg, consumed) = result.unwrap();
        assert_eq!(msg.msg_type, FixMsgType::Logon);
        assert_eq!(msg.sender_comp_id, "SENDER");
        assert_eq!(msg.target_comp_id, "TARGET");
        assert_eq!(msg.msg_seq_num, 1);
        assert!(consumed > 0);
    }

    #[test]
    fn test_parse_new_order_single() {
        let raw = make_fix_new_order();
        let mut parser = FixParser::new(FixVersion::Fix42);
        let (msg, _) = parser.parse_message(&raw).unwrap();
        assert_eq!(msg.msg_type, FixMsgType::NewOrderSingle);
        assert_eq!(msg.get(55), Some("AAPL"));
        assert_eq!(msg.order_side(), Some(BookSide::Bid));
        assert_eq!(msg.qty(), Some(100));
        let price = msg.get_f64(44).unwrap();
        assert!((price - 150.25).abs() < 0.001);
    }

    #[test]
    fn test_parse_exec_report() {
        let raw = make_fix_exec_report();
        let mut parser = FixParser::new(FixVersion::Fix42);
        let (msg, _) = parser.parse_message(&raw).unwrap();
        assert_eq!(msg.msg_type, FixMsgType::ExecutionReport);
        assert_eq!(msg.get(11), Some("ORDER001"));
    }

    #[test]
    fn test_fix_version_detection() {
        let fix44 = b"8=FIX.4.4\x019=10\x0135=0\x0149=A\x0156=B\x0134=1\x0152=T\x0110=0\x01";
        let mut parser = FixParser::new(FixVersion::Fix44);
        let (msg, _) = parser.parse_message(fix44).unwrap();
        assert_eq!(msg.version, FixVersion::Fix44);
    }

    #[test]
    fn test_fix_field_access_missing_tag() {
        let raw = b"8=FIX.4.2\x019=10\x0135=0\x0149=A\x0156=B\x0134=1\x0152=T\x0110=0\x01";
        let mut parser = FixParser::new(FixVersion::Fix42);
        let (msg, _) = parser.parse_message(raw).unwrap();
        assert_eq!(msg.get(999), None);
        assert_eq!(msg.get_f64(999), None);
    }

    #[test]
    fn test_fix_parse_file_empty() {
        let mut parser = FixParser::new(FixVersion::Fix42);
        // Parse empty bytes
        let result = parser.parse_message(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_fix_multiple_messages() {
        let mut raw = Vec::new();
        raw.extend_from_slice(b"8=FIX.4.2\x019=10\x0135=0\x0149=A\x0156=B\x0134=1\x0152=T\x0110=0\x01");
        raw.extend_from_slice(b"8=FIX.4.2\x019=10\x0135=5\x0149=A\x0156=B\x0134=2\x0152=T\x0110=0\x01");
        let mut parser = FixParser::new(FixVersion::Fix42);
        let (m1, c1) = parser.parse_message(&raw).unwrap();
        assert_eq!(m1.msg_type, FixMsgType::Heartbeat);
        let (m2, _) = parser.parse_message(&raw[c1..]).unwrap();
        assert_eq!(m2.msg_type, FixMsgType::Logout);
    }

    #[test]
    fn test_fix_stats() {
        let mut parser = FixParser::new(FixVersion::Fix42);
        let raw = make_fix_logon();
        let _ = parser.parse_message(&raw).unwrap();
        let _ = parser.parse_message(&raw).unwrap();
        let (n, _) = parser.stats();
        assert_eq!(n, 2);
    }

    #[test]
    fn test_fix_truncated_message() {
        let partial = b"8=FIX.4.2\x019=49\x0135=D\x0149=SEN";
        let mut parser = FixParser::new(FixVersion::Fix42);
        let result = parser.parse_message(partial);
        assert!(result.is_err());
    }

    #[test]
    fn test_fix_all_msg_types() {
        let tests = vec![
            ("0", FixMsgType::Heartbeat),
            ("A", FixMsgType::Logon),
            ("5", FixMsgType::Logout),
            ("D", FixMsgType::NewOrderSingle),
            ("8", FixMsgType::ExecutionReport),
            ("F", FixMsgType::OrderCancelRequest),
            ("V", FixMsgType::MarketDataRequest),
        ];
        for (code, expected) in tests {
            let raw = format!("8=FIX.4.2\x019=10\x0135={}\x0149=A\x0156=B\x0134=1\x0152=T\x0110=0\x01", code);
            let mut parser = FixParser::new(FixVersion::Fix42);
            let (msg, _) = parser.parse_message(raw.as_bytes()).unwrap();
            assert_eq!(msg.msg_type, expected, "failed for code {}", code);
        }
    }

    #[test]
    fn test_fix_order_side_sell() {
        let raw = b"8=FIX.4.2\x019=10\x0135=D\x0149=A\x0156=B\x0134=1\x0152=T\x0154=2\x0138=100\x0144=50.0\x0110=0\x01";
        let mut parser = FixParser::new(FixVersion::Fix42);
        let (msg, _) = parser.parse_message(raw).unwrap();
        assert_eq!(msg.order_side(), Some(BookSide::Ask));
    }
}

// ── ITCH parser tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod itch_tests {
    use super::*;

    fn make_itch_frame(msg_type: u8, payload: &[u8]) -> Vec<u8> {
        let len = payload.len() as u16;
        let mut frame = len.to_be_bytes().to_vec();
        frame.push(msg_type);
        frame.extend_from_slice(&payload[1..]); // skip type byte (already added)
        // pad to full length
        while frame.len() < 2 + payload.len() {
            frame.push(0);
        }
        frame
    }

    fn build_add_order_frame(ref_num: u64, side: u8, shares: u32, stock: &[u8; 8], price: u32) -> Vec<u8> {
        let mut payload = vec![0u8; 36];
        payload[0] = b'A';
        payload[11..19].copy_from_slice(&ref_num.to_be_bytes());
        payload[19] = side;
        payload[20..24].copy_from_slice(&shares.to_be_bytes());
        payload[24..32].copy_from_slice(stock);
        payload[32..36].copy_from_slice(&price.to_be_bytes());
        let len = payload.len() as u16;
        let mut frame = len.to_be_bytes().to_vec();
        frame.extend(payload);
        frame
    }

    fn build_delete_frame(ref_num: u64) -> Vec<u8> {
        let mut payload = vec![0u8; 19];
        payload[0] = b'D';
        payload[11..19].copy_from_slice(&ref_num.to_be_bytes());
        let len = payload.len() as u16;
        let mut frame = len.to_be_bytes().to_vec();
        frame.extend(payload);
        frame
    }

    #[test]
    fn test_parse_add_order() {
        let stock = b"TSLA    ";
        let frame = build_add_order_frame(99, b'B', 200, stock, 3000000);
        let mut parser = ItchParser::new();
        let (msg, consumed) = parser.parse_framed(&frame).unwrap();
        assert_eq!(consumed, 2 + 36);
        match msg {
            ItchMessage::AddOrder { ref_num, side, shares, price, .. } => {
                assert_eq!(ref_num, 99);
                assert_eq!(side, BookSide::Bid);
                assert_eq!(shares, 200);
                assert_eq!(price, 3000000);
            }
            other => panic!("Expected AddOrder, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_multiple_add_orders() {
        let stock = b"NVDA    ";
        let mut buffer = Vec::new();
        for i in 1u64..=10 {
            buffer.extend(build_add_order_frame(i, b'S', 100, stock, 500 * i as u32));
        }
        let mut parser = ItchParser::new();
        let messages = parser.parse_buffer(&buffer).unwrap();
        assert_eq!(messages.len(), 10, "expected 10 messages, got {}", messages.len());
        let (parsed, _) = parser.stats();
        assert_eq!(parsed, 10);
    }

    #[test]
    fn test_parse_order_delete() {
        let frame = build_delete_frame(42);
        let mut parser = ItchParser::new();
        let (msg, _) = parser.parse_framed(&frame).unwrap();
        match msg {
            ItchMessage::OrderDelete { ref_num } => assert_eq!(ref_num, 42),
            other => panic!("Expected Delete, got {:?}", other),
        }
    }

    #[test]
    fn test_itch_timestamp_parsing() {
        let stock = b"AMZN    ";
        let mut payload = vec![0u8; 36];
        payload[0] = b'A';
        // Set 6-byte timestamp at offset 5: value = 1_000_000_000 (1 second)
        let ts: u64 = 1_000_000_000;
        let ts_bytes: [u8; 8] = ts.to_be_bytes();
        payload[5..11].copy_from_slice(&ts_bytes[2..]); // 6 bytes
        payload[11..19].copy_from_slice(&1u64.to_be_bytes());
        payload[19] = b'B';
        payload[20..24].copy_from_slice(&100u32.to_be_bytes());
        payload[24..32].copy_from_slice(stock);
        payload[32..36].copy_from_slice(&100000u32.to_be_bytes());

        let len = payload.len() as u16;
        let mut frame = len.to_be_bytes().to_vec();
        frame.extend(payload);

        let mut parser = ItchParser::new();
        let (msg, _) = parser.parse_framed(&frame).unwrap();
        let ts = msg.timestamp();
        // Timestamp should be non-zero (values close to 1s)
        assert!(ts.0 > 0);
    }

    #[test]
    fn test_itch_unknown_type_skipped() {
        let mut parser = ItchParser::new(); // skip_unknown = true by default
        let mut buf = vec![0u8; 12]; // 2 len + 10 payload
        buf[0] = 0; buf[1] = 10; // length = 10
        buf[2] = 0xFF; // unknown type
        let result = parser.parse_framed(&buf);
        assert!(result.is_err()); // should be UnknownMessageType or skip
    }

    #[test]
    fn test_itch_book_reconstruction_depth() {
        let stock = *b"GOOG    ";
        let mut book = ReconstructedBook::new(stock);

        // Add multiple bid levels
        for i in 1u32..=5 {
            book.apply(&ItchMessage::AddOrder {
                timestamp: Nanos(i as u64 * 1000),
                ref_num: i as u64,
                side: BookSide::Bid,
                shares: 100 * i,
                stock,
                price: 1000 - i,
            });
        }
        // Add multiple ask levels
        for i in 1u32..=5 {
            book.apply(&ItchMessage::AddOrder {
                timestamp: Nanos(5000 + i as u64 * 1000),
                ref_num: 100 + i as u64,
                side: BookSide::Ask,
                shares: 100 * i,
                stock,
                price: 1001 + i,
            });
        }

        let (bids, asks) = book.depth_n_levels(5);
        assert_eq!(bids.len(), 5);
        assert_eq!(asks.len(), 5);
        // Bids should be sorted descending
        for i in 0..bids.len()-1 {
            assert!(bids[i].0 >= bids[i+1].0, "bids not sorted: {:?}", bids);
        }
        // Asks should be sorted ascending
        for i in 0..asks.len()-1 {
            assert!(asks[i].0 <= asks[i+1].0, "asks not sorted: {:?}", asks);
        }
    }

    #[test]
    fn test_itch_order_replace() {
        let stock = *b"META    ";
        let mut book = ReconstructedBook::new(stock);

        book.apply(&ItchMessage::AddOrder {
            timestamp: Nanos(1),
            ref_num: 1,
            side: BookSide::Bid,
            shares: 100,
            stock,
            price: 300,
        });

        book.apply(&ItchMessage::OrderReplace {
            timestamp: Nanos(2),
            orig_ref_num: 1,
            new_ref_num: 2,
            shares: 200,
            price: 301,
        });

        // Old price level 300 should be gone
        assert_eq!(book.bids.get(&300), None);
        // New price level 301 should exist
        assert!(book.bids.get(&301).is_some());
        assert_eq!(*book.bids.get(&301).unwrap(), 200);
    }

    #[test]
    fn test_multi_stock_reconstructor() {
        let mut reconstructor = BookReconstructor::new();
        let stocks = [*b"AAPL    ", *b"MSFT    ", *b"GOOG    "];

        for (i, &stock) in stocks.iter().enumerate() {
            let add = ItchMessage::AddOrder {
                timestamp: Nanos(i as u64 * 1000),
                ref_num: i as u64 * 100 + 1,
                side: BookSide::Bid,
                shares: 500,
                stock,
                price: 1000 * (i as u32 + 1),
            };
            reconstructor.apply(&add);
        }

        assert_eq!(reconstructor.book_count(), 3);
        for &stock in &stocks {
            let book = reconstructor.get_book(&stock);
            assert!(book.is_some(), "book not found for {:?}", std::str::from_utf8(&stock).unwrap_or("?"));
            assert_eq!(book.unwrap().order_count(), 1);
        }
    }

    #[test]
    fn test_book_trade_tracking() {
        let stock = *b"SPY     ";
        let mut book = ReconstructedBook::new(stock);

        book.apply(&ItchMessage::AddOrder {
            timestamp: Nanos(1),
            ref_num: 1,
            side: BookSide::Bid,
            shares: 1000,
            stock,
            price: 450,
        });
        book.apply(&ItchMessage::OrderExecuted {
            timestamp: Nanos(2),
            ref_num: 1,
            executed_shares: 300,
            match_number: 42,
        });

        assert_eq!(book.last_trade_qty, 300);
        assert_eq!(book.last_trade_price, Some(450));
        assert_eq!(book.total_trade_volume, 300);
    }
}

// ── PCAP reader tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod pcap_tests {
    use super::*;

    fn make_pcap_header(nanosecond: bool) -> Vec<u8> {
        let magic: u32 = if nanosecond { 0x4d3cb2a1 } else { 0xd4c3b2a1 };
        let mut h = Vec::new();
        h.extend(magic.to_le_bytes());
        h.extend(2u16.to_le_bytes()); // major
        h.extend(4u16.to_le_bytes()); // minor
        h.extend(0i32.to_le_bytes()); // thiszone
        h.extend(0u32.to_le_bytes()); // sigfigs
        h.extend(65535u32.to_le_bytes()); // snaplen
        h.extend(1u32.to_le_bytes()); // ethernet link type
        h
    }

    fn make_pcap_packet(ts_sec: u32, ts_frac: u32, data: &[u8]) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend(ts_sec.to_le_bytes());
        p.extend(ts_frac.to_le_bytes());
        let incl_len = data.len() as u32;
        p.extend(incl_len.to_le_bytes());
        p.extend(incl_len.to_le_bytes()); // orig_len = incl_len
        p.extend_from_slice(data);
        p
    }

    #[test]
    fn test_pcap_header_parse() {
        let header_bytes = make_pcap_header(false);
        let (reader, _consumed) = PcapReader::from_buffer(&header_bytes).unwrap();
        let h = reader.header();
        assert_eq!(h.major_version, 2);
        assert_eq!(h.minor_version, 4);
        assert_eq!(h.snap_len, 65535);
        assert_eq!(h.link_type, 1);
    }

    #[test]
    fn test_pcap_header_nanosecond() {
        let header_bytes = make_pcap_header(true);
        let (reader, _) = PcapReader::from_buffer(&header_bytes).unwrap();
        let _ = reader.header();
    }

    #[test]
    fn test_pcap_read_packets() {
        let mut buf = make_pcap_header(false);
        // Two packets
        buf.extend(make_pcap_packet(1704067200, 500000, &[0u8; 60]));
        buf.extend(make_pcap_packet(1704067200, 600000, &[0u8; 60]));

        let (reader, _) = PcapReader::from_buffer(&buf).unwrap();
        let packets = PcapReader::read_all_packets(&buf, reader.header()).unwrap();
        assert_eq!(packets.len(), 2);
        // First packet timestamp: 1704067200 sec + 500000 microseconds
        assert_eq!(packets[0].timestamp.0, 1704067200 * 1_000_000_000 + 500000 * 1000);
    }

    #[test]
    fn test_pcap_timestamp_nanosecond_resolution() {
        let mut buf = make_pcap_header(true); // nanosecond PCAP
        buf.extend(make_pcap_packet(100, 999999999, &[0u8; 14]));
        let (reader, _) = PcapReader::from_buffer(&buf).unwrap();
        let packets = PcapReader::read_all_packets(&buf, reader.header()).unwrap();
        assert_eq!(packets.len(), 1);
        // Nanosecond: ts = 100 * 1e9 + 999999999
        assert_eq!(packets[0].timestamp.0, 100_999_999_999);
    }

    #[test]
    fn test_pcap_empty_file() {
        let buf = make_pcap_header(false);
        let (reader, _) = PcapReader::from_buffer(&buf).unwrap();
        let packets = PcapReader::read_all_packets(&buf, reader.header()).unwrap();
        assert_eq!(packets.len(), 0);
    }

    #[test]
    fn test_pcap_invalid_magic() {
        let buf = [0xAA, 0xBB, 0xCC, 0xDD, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0];
        let result = PcapReader::from_buffer(&buf);
        assert!(result.is_err());
        match result.unwrap_err() {
            ReplayError::InvalidMagic { .. } => {}
            other => panic!("Expected InvalidMagic, got {:?}", other),
        }
    }

    #[test]
    fn test_udp_payload_extraction_too_short() {
        let frame = [0u8; 10];
        assert!(extract_udp_payload(&frame).is_none());
    }

    #[test]
    fn test_udp_payload_not_ipv4() {
        let mut frame = vec![0u8; 60];
        // Set EtherType to IPv6 (0x86DD)
        frame[12] = 0x86;
        frame[13] = 0xDD;
        assert!(extract_udp_payload(&frame).is_none());
    }
}

// ── Playback config tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod playback_tests {
    use super::*;

    #[test]
    fn test_turbo_mode() {
        let cfg = PlaybackConfig::turbo();
        assert!(cfg.turbo_mode);
        assert_eq!(cfg.speed, 1.0);
        assert!(cfg.start_time.is_none());
        assert!(cfg.end_time.is_none());
    }

    #[test]
    fn test_speed_config() {
        let cfg = PlaybackConfig::at_speed(10.0);
        assert_eq!(cfg.speed, 10.0);
        assert!(!cfg.turbo_mode);
    }

    #[test]
    fn test_time_filter() {
        let stock = *b"TEST    ";
        let mut packets = Vec::new();
        // Build synthetic ITCH packets
        for i in 0..10u64 {
            let ts = Nanos(i * 1_000_000_000);
            packets.push(crate::data_replay::PcapPacket {
                timestamp: ts,
                orig_len: 0,
                data: vec![0u8; 14], // minimal data
            });
        }

        let config = PlaybackConfig {
            turbo_mode: true,
            start_time: Some(Nanos(3_000_000_000)),
            end_time: Some(Nanos(6_000_000_000)),
            ..Default::default()
        };

        let mut session = PcapReplaySession::new(packets, config);
        let mut count = 0;
        while let Some(_) = session.next_itch_messages() {
            count += 1;
        }
        // Should process packets 3-6 (inclusive)
        assert!(count <= 4, "expected <= 4 packets, got {}", count);
    }

    #[test]
    fn test_max_messages_limit() {
        let stock = *b"TEST    ";
        let mut add_order_payload = vec![0u8; 36];
        add_order_payload[0] = b'A';
        add_order_payload[11..19].copy_from_slice(&1u64.to_be_bytes());
        add_order_payload[19] = b'B';
        add_order_payload[20..24].copy_from_slice(&100u32.to_be_bytes());
        add_order_payload[24..32].copy_from_slice(b"TEST    ");
        add_order_payload[32..36].copy_from_slice(&1000u32.to_be_bytes());

        let len = add_order_payload.len() as u16;
        let mut frame = len.to_be_bytes().to_vec();
        frame.extend(add_order_payload);

        // Minimal Ethernet + IP + UDP header
        let mut eth_ip_udp = vec![0u8; 42]; // 14+20+8
        eth_ip_udp[12] = 0x08; eth_ip_udp[13] = 0x00; // IPv4
        eth_ip_udp[23] = 17; // UDP
        eth_ip_udp[37] = 0x00; eth_ip_udp[38] = (frame.len() as u16 + 8) as u8; // UDP length

        let mut data = eth_ip_udp.clone();
        data.extend(&frame);

        let packets: Vec<_> = (0..20).map(|i| crate::data_replay::PcapPacket {
            timestamp: Nanos(i * 1_000_000),
            orig_len: data.len() as u32,
            data: data.clone(),
        }).collect();

        let config = PlaybackConfig {
            turbo_mode: true,
            max_messages: 5,
            ..Default::default()
        };

        let mut session = PcapReplaySession::new(packets, config);
        let mut total_msgs = 0u64;
        while let Some((_, msgs)) = session.next_itch_messages() {
            total_msgs += msgs.len() as u64;
            if session.msgs_emitted() >= 5 { break; }
        }
    }
}

// ── Nanos arithmetic tests ────────────────────────────────────────────────────

#[cfg(test)]
mod nanos_tests {
    use super::*;

    #[test]
    fn test_nanos_from_secs() { assert_eq!(Nanos::from_secs(1).0, 1_000_000_000); }
    #[test]
    fn test_nanos_from_millis() { assert_eq!(Nanos::from_millis(1).0, 1_000_000); }
    #[test]
    fn test_nanos_from_micros() { assert_eq!(Nanos::from_micros(1).0, 1_000); }
    #[test]
    fn test_nanos_as_secs_f64() { assert!((Nanos::from_secs(2).as_secs_f64() - 2.0).abs() < 1e-9); }
    #[test]
    fn test_nanos_add() { assert_eq!((Nanos(100) + Nanos(200)).0, 300); }
    #[test]
    fn test_nanos_sub_no_underflow() { assert_eq!((Nanos(100) - Nanos(200)).0, 0); }
    #[test]
    fn test_nanos_ordering() {
        assert!(Nanos(1000) > Nanos(500));
        assert!(Nanos(0) < Nanos(1));
        assert_eq!(Nanos(42), Nanos(42));
    }
    #[test]
    fn test_nanos_duration_since() {
        let a = Nanos(2000);
        let b = Nanos(1000);
        assert_eq!(a.duration_since(b).0, 1000);
    }
}
