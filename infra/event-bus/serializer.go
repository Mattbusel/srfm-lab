// serializer.go — JSON and MessagePack serializers for Event.
package eventbus

import (
	"encoding/json"
	"fmt"

	"github.com/vmihailenco/msgpack/v5"
)

// Serializer defines the interface for event serialization.
type Serializer interface {
	Marshal(evt *Event) ([]byte, error)
	Unmarshal(data []byte) (*Event, error)
}

// ─────────────────────────────────────────────────────────────────────────────
// JSONSerializer
// ─────────────────────────────────────────────────────────────────────────────

// JSONSerializer uses standard library JSON encoding.
type JSONSerializer struct{}

func (s *JSONSerializer) Marshal(evt *Event) ([]byte, error) {
	data, err := json.Marshal(evt)
	if err != nil {
		return nil, fmt.Errorf("json marshal: %w", err)
	}
	return data, nil
}

func (s *JSONSerializer) Unmarshal(data []byte) (*Event, error) {
	var evt Event
	if err := json.Unmarshal(data, &evt); err != nil {
		return nil, fmt.Errorf("json unmarshal: %w", err)
	}
	return &evt, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// MsgpackSerializer
// ─────────────────────────────────────────────────────────────────────────────

// MsgpackSerializer uses MessagePack for more compact, faster serialization.
type MsgpackSerializer struct{}

func (s *MsgpackSerializer) Marshal(evt *Event) ([]byte, error) {
	data, err := msgpack.Marshal(evt)
	if err != nil {
		return nil, fmt.Errorf("msgpack marshal: %w", err)
	}
	return data, nil
}

func (s *MsgpackSerializer) Unmarshal(data []byte) (*Event, error) {
	var evt Event
	if err := msgpack.Unmarshal(data, &evt); err != nil {
		// Fallback: try JSON (in case the message was published by a JSON publisher).
		var jsonEvt Event
		if jsonErr := json.Unmarshal(data, &jsonEvt); jsonErr == nil {
			return &jsonEvt, nil
		}
		return nil, fmt.Errorf("msgpack unmarshal: %w", err)
	}
	return &evt, nil
}
