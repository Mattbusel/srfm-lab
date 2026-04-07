package main

import (
	"bufio"
	"bytes"
	"fmt"
	"log"
	"math"
	"net"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// SessionState enumerates the FIX session state machine states.
type SessionState int

const (
	StateDisconnected SessionState = iota
	StateConnected
	StateLogonSent
	StateActive
	StateLogoutSent
)

// String returns a human-readable name for the session state.
func (s SessionState) String() string {
	switch s {
	case StateDisconnected:
		return "DISCONNECTED"
	case StateConnected:
		return "CONNECTED"
	case StateLogonSent:
		return "LOGON_SENT"
	case StateActive:
		return "ACTIVE"
	case StateLogoutSent:
		return "LOGOUT_SENT"
	}
	return "UNKNOWN"
}

// FIXConfig holds all parameters required to establish a FIX session.
type FIXConfig struct {
	SenderCompID      string
	TargetCompID      string
	Host              string
	Port              int
	HeartbeatInterval int  // seconds
	ResetOnLogon      bool
	BeginString       string // e.g. "FIX.4.2"
}

// FIXOrder is the internal representation of an order to be sent via FIX.
type FIXOrder struct {
	Symbol  string
	Side    string // "1"=buy "2"=sell
	Qty     string // string to avoid float formatting issues
	Price   string // "" for market orders
	OrdType string // "1"=market "2"=limit
}

// FIXSession manages a single FIX 4.2 session over TCP.
type FIXSession struct {
	cfg FIXConfig

	mu    sync.Mutex
	state SessionState

	conn      net.Conn
	reader    *bufio.Reader

	seqNumOut int64 // next outgoing seq num (atomic)
	seqNumIn  int64 // expected incoming seq num

	heartbeatTicker *time.Ticker
	done            chan struct{}

	// onExecutionReport is called for every received 35=8 message.
	onExecutionReport func(ExecutionReport)

	// reconnect backoff
	reconnectAttempt int
}

// NewFIXSession creates a FIXSession. Call Connect to establish the TCP link.
func NewFIXSession(cfg FIXConfig) *FIXSession {
	if cfg.HeartbeatInterval <= 0 {
		cfg.HeartbeatInterval = 30
	}
	if cfg.BeginString == "" {
		cfg.BeginString = "FIX.4.2"
	}
	s := &FIXSession{
		cfg:       cfg,
		state:     StateDisconnected,
		seqNumOut: 1,
		seqNumIn:  1,
		done:      make(chan struct{}),
	}
	return s
}

// SetExecutionReportHandler registers a callback for execution reports.
func (s *FIXSession) SetExecutionReportHandler(fn func(ExecutionReport)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.onExecutionReport = fn
}

// State returns the current session state.
func (s *FIXSession) State() SessionState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state
}

// IsActive returns true when the session is in StateActive.
func (s *FIXSession) IsActive() bool {
	return s.State() == StateActive
}

// SeqNumOut returns the next outgoing sequence number.
func (s *FIXSession) SeqNumOut() int64 {
	return atomic.LoadInt64(&s.seqNumOut)
}

// SeqNumIn returns the expected incoming sequence number.
func (s *FIXSession) SeqNumIn() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.seqNumIn
}

// Connect establishes a TCP connection and sends the Logon message.
// It starts the read loop and heartbeat scheduler.
func (s *FIXSession) Connect() error {
	addr := fmt.Sprintf("%s:%d", s.cfg.Host, s.cfg.Port)
	conn, err := net.DialTimeout("tcp", addr, 10*time.Second)
	if err != nil {
		return fmt.Errorf("tcp dial %s: %w", addr, err)
	}

	s.mu.Lock()
	s.conn = conn
	s.reader = bufio.NewReader(conn)
	s.state = StateConnected
	s.reconnectAttempt = 0
	s.mu.Unlock()

	log.Printf("fix-session: connected to %s", addr)

	if err := s.sendLogon(); err != nil {
		conn.Close()
		return err
	}

	go s.readLoop()
	go s.heartbeatLoop()

	return nil
}

// ConnectWithRetry connects to the counterparty and auto-reconnects on disconnect
// using exponential backoff. It runs until Close is called.
func (s *FIXSession) ConnectWithRetry() {
	for {
		select {
		case <-s.done:
			return
		default:
		}

		err := s.Connect()
		if err != nil {
			wait := backoff(s.reconnectAttempt)
			log.Printf("fix-session: connect error: %v, retrying in %v", err, wait)
			s.mu.Lock()
			s.reconnectAttempt++
			s.mu.Unlock()

			select {
			case <-time.After(wait):
			case <-s.done:
				return
			}
		}
	}
}

// backoff returns exponential wait capped at 60s.
func backoff(attempt int) time.Duration {
	base := 1.0
	max := 60.0
	d := base * math.Pow(2, float64(attempt))
	if d > max {
		d = max
	}
	return time.Duration(d * float64(time.Second))
}

// Close sends Logout and closes the TCP connection.
func (s *FIXSession) Close() error {
	close(s.done)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn != nil {
		_ = s.sendLogoutLocked("normal logout")
		s.conn.Close()
		s.conn = nil
	}
	s.state = StateDisconnected
	return nil
}

// SendOrder encodes and transmits a NewOrderSingle. Returns the ClOrdID.
func (s *FIXSession) SendOrder(order FIXOrder) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state != StateActive {
		return "", fmt.Errorf("session not active (state=%s)", s.state)
	}

	clOrdID := generateClOrdID()
	transactTime := time.Now().UTC().Format("20060102-15:04:05")

	msg := NewOrderSingle(clOrdID, order.Symbol, order.Side, order.Qty, order.Price, order.OrdType, transactTime)
	if err := s.sendMsgLocked(msg); err != nil {
		return "", err
	}
	return clOrdID, nil
}

// CancelOrder sends an OrderCancelRequest for the given original ClOrdID.
func (s *FIXSession) CancelOrder(origClOrdID, symbol, side string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state != StateActive {
		return fmt.Errorf("session not active (state=%s)", s.state)
	}

	clOrdID := generateClOrdID()
	transactTime := time.Now().UTC().Format("20060102-15:04:05")
	msg := NewOrderCancelRequest(clOrdID, origClOrdID, symbol, side, transactTime)
	return s.sendMsgLocked(msg)
}

// -- internal send helpers --

func (s *FIXSession) sendLogon() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	msg := NewLogon(s.cfg.HeartbeatInterval, s.cfg.ResetOnLogon)
	if err := s.sendMsgLocked(msg); err != nil {
		return err
	}
	s.state = StateLogonSent
	return nil
}

func (s *FIXSession) sendLogoutLocked(text string) error {
	msg := NewLogout(text)
	if err := s.sendMsgLocked(msg); err != nil {
		return err
	}
	s.state = StateLogoutSent
	return nil
}

// sendMsgLocked stamps the session header fields and writes to TCP.
// Caller must hold s.mu.
func (s *FIXSession) sendMsgLocked(msg FIXMessage) error {
	seq := atomic.AddInt64(&s.seqNumOut, 1) - 1

	msg.Set(TagSenderCompID, s.cfg.SenderCompID)
	msg.Set(TagTargetCompID, s.cfg.TargetCompID)
	msg.Set(TagMsgSeqNum, strconv.FormatInt(seq, 10))
	msg.Set(TagSendingTime, time.Now().UTC().Format("20060102-15:04:05.000"))

	wire := Encode(msg)

	if s.conn == nil {
		return fmt.Errorf("no connection")
	}
	_, err := s.conn.Write(wire)
	if err != nil {
		s.state = StateDisconnected
		return fmt.Errorf("write: %w", err)
	}
	return nil
}

// -- read loop --

// readLoop reads FIX messages from TCP and dispatches them.
func (s *FIXSession) readLoop() {
	for {
		select {
		case <-s.done:
			return
		default:
		}

		msg, err := s.readMessage()
		if err != nil {
			log.Printf("fix-session: read error: %v", err)
			s.handleDisconnect()
			return
		}

		s.dispatch(msg)
	}
}

// readMessage reads exactly one FIX message from the buffered reader.
// It uses the BodyLength tag to know how many bytes to read.
func (s *FIXSession) readMessage() (FIXMessage, error) {
	s.mu.Lock()
	rdr := s.reader
	s.mu.Unlock()

	if rdr == nil {
		return FIXMessage{}, fmt.Errorf("no reader")
	}

	// Read until we accumulate a full FIX message.
	// Strategy: read fields until we have BodyLength and have consumed
	// that many bytes beyond the 9= field, then validate 10=.
	var buf bytes.Buffer

	// Read fields one at a time (each ends with SOH).
	bodyLength := -1
	headerBytes := 0
	bodyBytes := 0

	for {
		field, err := rdr.ReadString('\x01')
		if err != nil {
			return FIXMessage{}, err
		}
		buf.WriteString(field)

		eqIdx := bytes.IndexByte([]byte(field), '=')
		if eqIdx < 1 {
			continue
		}
		tagStr := field[:eqIdx]
		value := field[eqIdx+1 : len(field)-1] // strip trailing SOH

		tag, _ := strconv.Atoi(tagStr)

		if tag == TagBeginString || tag == TagBodyLength || tag == TagCheckSum {
			if tag == TagBodyLength {
				bodyLength, _ = strconv.Atoi(value)
				// headerBytes = bytes consumed for fields 8 and 9
				headerBytes = len("8=FIX.4.2" + SOH + field)
				_ = headerBytes
			}
			if tag == TagCheckSum {
				// Done.
				break
			}
			continue
		}

		if bodyLength >= 0 {
			bodyBytes += len(field)
			if bodyBytes >= bodyLength {
				// Read checksum field
				checksumField, err := rdr.ReadString('\x01')
				if err != nil {
					return FIXMessage{}, err
				}
				buf.WriteString(checksumField)
				break
			}
		}
	}

	return Decode(buf.Bytes())
}

// dispatch routes an incoming FIX message to the appropriate handler.
func (s *FIXSession) dispatch(msg FIXMessage) {
	s.mu.Lock()
	// Validate incoming seq num (simple gap detection).
	inSeq, _ := strconv.ParseInt(msg.Get(TagMsgSeqNum), 10, 64)
	if inSeq > 0 && inSeq > s.seqNumIn {
		log.Printf("fix-session: gap detected: expected %d got %d", s.seqNumIn, inSeq)
		// Respond with ResendRequest.
		go s.sendResendRequest(s.seqNumIn, inSeq-1)
	}
	if inSeq > 0 {
		s.seqNumIn = inSeq + 1
	}
	state := s.state
	onExec := s.onExecutionReport
	s.mu.Unlock()

	switch msg.MsgType {
	case MsgTypeLogon:
		s.mu.Lock()
		if state == StateLogonSent {
			s.state = StateActive
			log.Printf("fix-session: session active")
		}
		s.mu.Unlock()

	case MsgTypeLogout:
		log.Printf("fix-session: received Logout: %s", msg.Get(TagText))
		s.handleDisconnect()

	case MsgTypeHeartbeat:
		// Nothing to do -- heartbeat received, session is alive.

	case MsgTypeTestRequest:
		// Echo back a heartbeat with the same TestReqID.
		testReqID := msg.Get(TagTestReqID)
		hb := NewHeartbeat(testReqID)
		s.mu.Lock()
		_ = s.sendMsgLocked(hb)
		s.mu.Unlock()

	case MsgTypeExecutionReport:
		if onExec != nil {
			report := ParseExecutionReport(msg)
			go onExec(report)
		}

	default:
		log.Printf("fix-session: unhandled MsgType=%s", msg.MsgType)
	}
}

// handleDisconnect tears down the connection and resets state.
func (s *FIXSession) handleDisconnect() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn != nil {
		s.conn.Close()
		s.conn = nil
		s.reader = nil
	}
	s.state = StateDisconnected
	log.Printf("fix-session: disconnected")
}

// sendResendRequest sends a FIX ResendRequest for sequence range [beginSeq, endSeq].
func (s *FIXSession) sendResendRequest(beginSeq, endSeq int64) {
	msg := FIXMessage{Fields: make(map[int]string)}
	msg.Set(TagMsgType, MsgTypeResendRequest)
	msg.Set(7 /* BeginSeqNo */, strconv.FormatInt(beginSeq, 10))
	msg.Set(16 /* EndSeqNo */, strconv.FormatInt(endSeq, 10))
	s.mu.Lock()
	_ = s.sendMsgLocked(msg)
	s.mu.Unlock()
}

// -- heartbeat loop --

func (s *FIXSession) heartbeatLoop() {
	interval := time.Duration(s.cfg.HeartbeatInterval) * time.Second
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-s.done:
			return
		case <-ticker.C:
			s.mu.Lock()
			if s.state == StateActive {
				hb := NewHeartbeat("")
				if err := s.sendMsgLocked(hb); err != nil {
					log.Printf("fix-session: heartbeat send error: %v", err)
				}
			}
			s.mu.Unlock()
		}
	}
}

// generateClOrdID returns a unique order identifier based on current nanosecond time.
func generateClOrdID() string {
	return fmt.Sprintf("ORD%d", time.Now().UnixNano())
}
