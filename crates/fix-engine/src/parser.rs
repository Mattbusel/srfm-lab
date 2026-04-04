use crate::message::{FixMessage, FixField, FixTag, SOH, MessageError, RepeatingGroup, tags};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Empty input")]
    EmptyInput,
    #[error("Missing BeginString (tag 8) as first field")]
    MissingBeginString,
    #[error("Missing BodyLength (tag 9) as second field")]
    MissingBodyLength,
    #[error("Missing MsgType (tag 35)")]
    MissingMsgType,
    #[error("Malformed field at byte {0}: {1}")]
    MalformedField(usize, String),
    #[error("Invalid tag number: {0}")]
    InvalidTag(String),
    #[error("Truncated message: expected {expected} body bytes, got {got}")]
    Truncated { expected: usize, got: usize },
    #[error("Missing checksum (tag 10)")]
    MissingChecksum,
    #[error("Message error: {0}")]
    MessageError(#[from] MessageError),
    #[error("UTF-8 error")]
    Utf8Error,
}

/// Zero-copy FIX parser. Borrows from input buffer where possible.
pub struct FixParser {
    /// Known repeating group delimiter tags keyed by count tag
    group_delimiters: std::collections::HashMap<FixTag, FixTag>,
}

impl FixParser {
    pub fn new() -> Self {
        let mut group_delimiters = std::collections::HashMap::new();
        // NoMDEntryTypes (267) -> MDEntryType (269)
        group_delimiters.insert(tags::NoMDEntryTypes, tags::MDEntryType);
        // NoMDEntries (268) -> MDEntryType (269)
        group_delimiters.insert(tags::NoMDEntries, tags::MDEntryType);
        // NoRelatedSym (146) -> Symbol (55)
        group_delimiters.insert(tags::NoRelatedSym, tags::Symbol);
        FixParser { group_delimiters }
    }

    pub fn register_group_delimiter(&mut self, count_tag: FixTag, delimiter_tag: FixTag) {
        self.group_delimiters.insert(count_tag, delimiter_tag);
    }

    /// Parse a single complete FIX message from a byte slice.
    /// Returns (message, bytes_consumed).
    pub fn parse<'a>(&self, input: &'a [u8]) -> Result<(FixMessage, usize), ParseError> {
        if input.is_empty() {
            return Err(ParseError::EmptyInput);
        }

        // Split all fields
        let mut fields: Vec<(u32, &'a [u8])> = Vec::new();
        let mut pos = 0;
        let mut total_consumed = 0;

        while pos < input.len() {
            // Find '='
            let eq_pos = match input[pos..].iter().position(|&b| b == b'=') {
                Some(p) => pos + p,
                None => break,
            };
            let tag_bytes = &input[pos..eq_pos];
            let tag_str = std::str::from_utf8(tag_bytes).map_err(|_| ParseError::Utf8Error)?;
            let tag: u32 = tag_str.parse().map_err(|_| ParseError::InvalidTag(tag_str.to_string()))?;

            let val_start = eq_pos + 1;
            let soh_pos = match input[val_start..].iter().position(|&b| b == SOH) {
                Some(p) => val_start + p,
                None => {
                    // No more SOH — incomplete
                    break;
                }
            };
            let value = &input[val_start..soh_pos];
            fields.push((tag, value));

            pos = soh_pos + 1;
            total_consumed = pos;

            // If we found CheckSum (10), we're done
            if tag == tags::CheckSum {
                break;
            }
        }

        if fields.is_empty() {
            return Err(ParseError::EmptyInput);
        }

        // Validate structure
        if fields[0].0 != tags::BeginString {
            return Err(ParseError::MissingBeginString);
        }
        if fields.len() < 2 || fields[1].0 != tags::BodyLength {
            return Err(ParseError::MissingBodyLength);
        }

        // Require CheckSum (tag 10) to declare message complete
        let has_checksum = fields.iter().any(|(t, _)| *t == tags::CheckSum);
        if !has_checksum {
            return Err(ParseError::MissingChecksum);
        }

        let begin_string = std::str::from_utf8(fields[0].1).map_err(|_| ParseError::Utf8Error)?.to_string();
        let body_length: usize = std::str::from_utf8(fields[1].1)
            .map_err(|_| ParseError::Utf8Error)?
            .parse()
            .map_err(|_| ParseError::MalformedField(0, "BodyLength not integer".into()))?;

        // Find MsgType
        let msg_type_field = fields.iter()
            .find(|(t, _)| *t == tags::MsgType)
            .ok_or(ParseError::MissingMsgType)?;
        let msg_type = std::str::from_utf8(msg_type_field.1).map_err(|_| ParseError::Utf8Error)?.to_string();

        let mut message = FixMessage::new(&begin_string, &msg_type);

        for (tag, value) in &fields {
            message.set_field(FixField { tag: *tag, value: value.to_vec() });
        }

        Ok((message, total_consumed))
    }

    /// Parse repeating group fields from a slice starting after the count field.
    /// delimiter_tag is the first tag of each group instance.
    pub fn parse_repeating_group(
        &self,
        fields: &[(u32, &[u8])],
        count: usize,
        delimiter_tag: FixTag,
    ) -> RepeatingGroup {
        let mut group = RepeatingGroup::new(delimiter_tag);
        let mut current_instance: Vec<FixField> = Vec::new();
        let mut found = 0;

        for (tag, value) in fields {
            if *tag == delimiter_tag {
                if !current_instance.is_empty() {
                    group.add_instance(current_instance.clone());
                    current_instance.clear();
                    found += 1;
                    if found >= count {
                        break;
                    }
                }
                current_instance.push(FixField { tag: *tag, value: value.to_vec() });
            } else if !current_instance.is_empty() {
                current_instance.push(FixField { tag: *tag, value: value.to_vec() });
            }
        }

        if !current_instance.is_empty() && found < count {
            group.add_instance(current_instance);
        }

        group
    }

    /// Parse multiple messages from a streaming buffer.
    /// Returns parsed messages and the number of bytes consumed.
    pub fn parse_stream(&self, input: &[u8]) -> (Vec<FixMessage>, usize) {
        let mut messages = Vec::new();
        let mut offset = 0;

        loop {
            if offset >= input.len() {
                break;
            }
            match self.parse(&input[offset..]) {
                Ok((msg, consumed)) => {
                    messages.push(msg);
                    offset += consumed;
                    if consumed == 0 {
                        break;
                    }
                }
                Err(ParseError::EmptyInput) => break,
                Err(_) => break,
            }
        }

        (messages, offset)
    }

    /// Extract a single field value from raw wire bytes without full parse.
    /// Useful for quick tag extraction (e.g., peek at MsgType).
    pub fn extract_field<'a>(input: &'a [u8], tag: FixTag) -> Option<&'a [u8]> {
        let tag_str = tag.to_string();
        let tag_bytes = tag_str.as_bytes();
        let mut pos = 0;
        while pos < input.len() {
            let eq_pos = input[pos..].iter().position(|&b| b == b'=')?;
            let field_tag_bytes = &input[pos..pos + eq_pos];
            let val_start = pos + eq_pos + 1;
            let soh_pos = input[val_start..].iter().position(|&b| b == SOH)?;
            if field_tag_bytes == tag_bytes {
                return Some(&input[val_start..val_start + soh_pos]);
            }
            pos = val_start + soh_pos + 1;
        }
        None
    }
}

impl Default for FixParser {
    fn default() -> Self {
        Self::new()
    }
}

/// A stateful streaming parser that buffers incomplete messages
pub struct StreamingParser {
    inner: FixParser,
    buffer: Vec<u8>,
}

impl StreamingParser {
    pub fn new() -> Self {
        StreamingParser {
            inner: FixParser::new(),
            buffer: Vec::with_capacity(4096),
        }
    }

    /// Feed bytes into the parser, returns any complete messages parsed
    pub fn feed(&mut self, data: &[u8]) -> Vec<Result<FixMessage, ParseError>> {
        self.buffer.extend_from_slice(data);
        let mut results = Vec::new();
        let mut offset = 0;

        loop {
            if offset >= self.buffer.len() {
                break;
            }
            match self.inner.parse(&self.buffer[offset..]) {
                Ok((msg, consumed)) => {
                    if consumed == 0 {
                        break;
                    }
                    results.push(Ok(msg));
                    offset += consumed;
                }
                Err(ParseError::EmptyInput) => break,
                Err(ParseError::Truncated { .. }) => break,
                Err(ParseError::MissingChecksum) => break, // incomplete message, wait for more data
                Err(e) => {
                    results.push(Err(e));
                    // Skip past this malformed data — try to find next BeginString
                    let skip = self.buffer[offset..].windows(3)
                        .skip(1)
                        .position(|w| w == b"8=F")
                        .map(|p| p + 1)
                        .unwrap_or(self.buffer.len() - offset);
                    offset += skip;
                }
            }
        }

        // Drain consumed bytes
        self.buffer.drain(..offset);
        results
    }

    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    pub fn clear_buffer(&mut self) {
        self.buffer.clear();
    }
}

impl Default for StreamingParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::FixMessage;
    use crate::types::MsgType;

    fn make_wire_heartbeat() -> Vec<u8> {
        let mut msg = FixMessage::with_header("FIX.4.2", MsgType::Heartbeat.as_str(), "SENDER", "TARGET", 1);
        msg.encode()
    }

    #[test]
    fn test_parse_heartbeat_roundtrip() {
        let wire = make_wire_heartbeat();
        let parser = FixParser::new();
        let (msg, consumed) = parser.parse(&wire).expect("parse should succeed");
        assert_eq!(consumed, wire.len());
        assert_eq!(msg.msg_type, "0");
        assert_eq!(msg.begin_string, "FIX.4.2");
    }

    #[test]
    fn test_extract_field() {
        let wire = make_wire_heartbeat();
        let val = FixParser::extract_field(&wire, tags::SenderCompID)
            .and_then(|v| std::str::from_utf8(v).ok());
        assert_eq!(val, Some("SENDER"));
    }

    #[test]
    fn test_streaming_two_messages() {
        let wire1 = make_wire_heartbeat();
        let wire2 = make_wire_heartbeat();
        let mut combined = wire1.clone();
        combined.extend_from_slice(&wire2);

        let mut parser = StreamingParser::new();
        let results = parser.feed(&combined);
        assert_eq!(results.len(), 2);
        for r in results {
            assert!(r.is_ok());
        }
        assert_eq!(parser.buffer_len(), 0);
    }

    #[test]
    fn test_streaming_partial_feed() {
        let wire = make_wire_heartbeat();
        let mid = wire.len() / 2;
        let mut parser = StreamingParser::new();
        let r1 = parser.feed(&wire[..mid]);
        assert!(r1.is_empty()); // incomplete
        let r2 = parser.feed(&wire[mid..]);
        assert_eq!(r2.len(), 1);
        assert!(r2[0].is_ok());
    }

    #[test]
    fn test_parse_repeating_group() {
        let parser = FixParser::new();
        let fields: Vec<(u32, &[u8])> = vec![
            (tags::MDEntryType, b"0"),
            (tags::MDEntryPx, b"100.00"),
            (tags::MDEntryType, b"1"),
            (tags::MDEntryPx, b"100.05"),
        ];
        let group = parser.parse_repeating_group(&fields, 2, tags::MDEntryType);
        assert_eq!(group.count(), 2);
    }

    #[test]
    fn test_missing_begin_string() {
        // Craft a message starting without tag 8
        let bad: Vec<u8> = b"9=5\x0135=D\x0110=000\x01".to_vec();
        let parser = FixParser::new();
        assert!(matches!(parser.parse(&bad), Err(ParseError::MissingBeginString)));
    }
}
