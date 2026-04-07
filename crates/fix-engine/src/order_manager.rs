// order_manager.rs -- FIX order management state machine for SRFM.
// Tracks lifecycle of all submitted orders through ExecutionReport messages.

use std::collections::HashMap;
use thiserror::Error;
use crate::message::{FixMessage, tags};
use crate::types::ExecType;

// ---------------------------------------------------------------------------
// FIXError
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum FIXError {
    #[error("Missing field tag {0}")]
    MissingField(u32),
    #[error("Order not found: {0}")]
    OrderNotFound(String),
    #[error("Invalid state transition for order {cl_ord_id}: {msg}")]
    InvalidTransition { cl_ord_id: String, msg: String },
    #[error("Field parse error: {0}")]
    FieldError(#[from] crate::message::MessageError),
    #[error("Duplicate ClOrdID: {0}")]
    DuplicateOrder(String),
}

// ---------------------------------------------------------------------------
// OrderState
// ---------------------------------------------------------------------------

/// State machine for a single managed order.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderState {
    New,
    PendingNew,
    PartiallyFilled { filled_qty: f64, avg_px: f64 },
    Filled { filled_qty: f64, avg_px: f64 },
    PendingCancel,
    Cancelled,
    Rejected { reason: String },
}

impl OrderState {
    /// True if the order is in a terminal state and will receive no further updates.
    pub fn is_terminal(&self) -> bool {
        matches!(self, OrderState::Filled { .. } | OrderState::Cancelled | OrderState::Rejected { .. })
    }

    pub fn is_active(&self) -> bool {
        !self.is_terminal()
    }
}

impl std::fmt::Display for OrderState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderState::New => write!(f, "New"),
            OrderState::PendingNew => write!(f, "PendingNew"),
            OrderState::PartiallyFilled { filled_qty, avg_px } => {
                write!(f, "PartiallyFilled(qty={filled_qty:.4}, avg_px={avg_px:.4})")
            }
            OrderState::Filled { filled_qty, avg_px } => {
                write!(f, "Filled(qty={filled_qty:.4}, avg_px={avg_px:.4})")
            }
            OrderState::PendingCancel => write!(f, "PendingCancel"),
            OrderState::Cancelled => write!(f, "Cancelled"),
            OrderState::Rejected { reason } => write!(f, "Rejected({reason})"),
        }
    }
}

// ---------------------------------------------------------------------------
// OrderEvent
// ---------------------------------------------------------------------------

/// Event produced by the order manager after processing an ExecutionReport.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderEvent {
    Accepted,
    PartialFill { qty: f64, price: f64 },
    FullFill { qty: f64, price: f64 },
    CancelAck,
    Reject { reason: String },
    StateUnchanged,
}

// ---------------------------------------------------------------------------
// ManagedOrder
// ---------------------------------------------------------------------------

/// A single order tracked by FIXOrderManager.
#[derive(Debug, Clone)]
pub struct ManagedOrder {
    pub cl_ord_id: String,
    /// Exchange-assigned order ID, set on first ExecutionReport.
    pub order_id: Option<String>,
    pub symbol: String,
    /// FIX Side char: '1'=Buy, '2'=Sell, etc.
    pub side: char,
    /// Original submitted quantity.
    pub qty: f64,
    /// Limit price if applicable.
    pub price: Option<f64>,
    pub state: OrderState,
    /// Unix-nanosecond submission timestamp.
    pub submitted_at: i64,
}

impl ManagedOrder {
    pub fn new(
        cl_ord_id: impl Into<String>,
        symbol: impl Into<String>,
        side: char,
        qty: f64,
        price: Option<f64>,
        submitted_at: i64,
    ) -> Self {
        Self {
            cl_ord_id: cl_ord_id.into(),
            order_id: None,
            symbol: symbol.into(),
            side,
            qty,
            price,
            state: OrderState::PendingNew,
            submitted_at,
        }
    }

    pub fn filled_qty(&self) -> f64 {
        match &self.state {
            OrderState::PartiallyFilled { filled_qty, .. } => *filled_qty,
            OrderState::Filled { filled_qty, .. } => *filled_qty,
            _ => 0.0,
        }
    }

    pub fn avg_fill_price(&self) -> Option<f64> {
        match &self.state {
            OrderState::PartiallyFilled { avg_px, .. } => Some(*avg_px),
            OrderState::Filled { avg_px, .. } => Some(*avg_px),
            _ => None,
        }
    }

    pub fn is_done(&self) -> bool {
        self.state.is_terminal()
    }
}

// ---------------------------------------------------------------------------
// FIXOrderManager
// ---------------------------------------------------------------------------

/// Tracks all managed FIX orders and processes ExecutionReport messages.
#[derive(Default)]
pub struct FIXOrderManager {
    /// Primary index by ClOrdID.
    orders: HashMap<String, ManagedOrder>,
    /// Secondary index: exchange OrderID -> ClOrdID (populated after first ER).
    order_id_map: HashMap<String, String>,
    /// Running tally of submitted quantity (for fill_rate calculation).
    total_submitted_qty: f64,
    /// Running tally of filled quantity.
    total_filled_qty: f64,
}

impl FIXOrderManager {
    pub fn new() -> Self {
        Self::default()
    }

    // ---------------------------------------------------------------------------
    // Order registration
    // ---------------------------------------------------------------------------

    /// Register a new order before any ExecutionReport is received.
    /// Call this immediately after sending NewOrderSingle.
    pub fn register_order(
        &mut self,
        cl_ord_id: impl Into<String>,
        symbol: impl Into<String>,
        side: char,
        qty: f64,
        price: Option<f64>,
        submitted_at: i64,
    ) -> Result<(), FIXError> {
        let id: String = cl_ord_id.into();
        if self.orders.contains_key(&id) {
            return Err(FIXError::DuplicateOrder(id));
        }
        let order = ManagedOrder::new(id.clone(), symbol, side, qty, price, submitted_at);
        self.total_submitted_qty += qty;
        self.orders.insert(id, order);
        Ok(())
    }

    // ---------------------------------------------------------------------------
    // on_execution_report
    // ---------------------------------------------------------------------------

    /// Process an incoming ExecutionReport message and advance order state.
    ///
    /// The message must contain tag 150 (ExecType) and tag 11 (ClOrdID).
    /// Returns the event representing the state change.
    pub fn on_execution_report(&mut self, msg: &FixMessage) -> Result<OrderEvent, FIXError> {
        let cl_ord_id = msg
            .get_field(tags::ClOrdID)
            .ok_or(FIXError::MissingField(tags::ClOrdID))?
            .value_str()?
            .to_string();

        let exec_type_str = msg
            .get_field(tags::ExecType)
            .ok_or(FIXError::MissingField(tags::ExecType))?
            .value_str()?;

        let exec_type = ExecType::from_fix_char(exec_type_str.as_bytes().first().copied().unwrap_or(b'I'))
            .unwrap_or(ExecType::OrderStatus);

        // Look up order -- also try OrigClOrdID for cancel confirmations
        let resolved_id = if self.orders.contains_key(&cl_ord_id) {
            cl_ord_id.clone()
        } else if let Some(orig) = msg.get_field(tags::OrigClOrdID)
            .and_then(|f| f.value_str().ok())
        {
            if self.orders.contains_key(orig) {
                orig.to_string()
            } else {
                return Err(FIXError::OrderNotFound(cl_ord_id));
            }
        } else {
            return Err(FIXError::OrderNotFound(cl_ord_id));
        };

        // Update exchange OrderID if provided
        if let Some(oid_field) = msg.get_field(tags::OrderID) {
            if let Ok(oid) = oid_field.value_str() {
                if !oid.is_empty() && oid != "NONE" {
                    self.order_id_map.insert(oid.to_string(), resolved_id.clone());
                    if let Some(order) = self.orders.get_mut(&resolved_id) {
                        if order.order_id.is_none() {
                            order.order_id = Some(oid.to_string());
                        }
                    }
                }
            }
        }

        let event = match exec_type {
            ExecType::New | ExecType::PendingNew => {
                if let Some(order) = self.orders.get_mut(&resolved_id) {
                    if order.state == OrderState::PendingNew {
                        order.state = OrderState::New;
                    }
                }
                OrderEvent::Accepted
            }

            ExecType::PartialFill | ExecType::Trade => {
                let last_qty = msg.get_field(tags::LastQty)
                    .ok_or(FIXError::MissingField(tags::LastQty))?
                    .value_f64()?;
                let last_px = msg.get_field(tags::LastPx)
                    .ok_or(FIXError::MissingField(tags::LastPx))?
                    .value_f64()?;
                let cum_qty = msg.get_field(tags::CumQty)
                    .and_then(|f| f.value_f64().ok())
                    .unwrap_or(last_qty);
                let avg_px = msg.get_field(tags::AvgPx)
                    .and_then(|f| f.value_f64().ok())
                    .unwrap_or(last_px);

                // Check OrdStatus to determine if this is actually a full fill
                let ord_status = msg.get_field(tags::OrdStatus)
                    .and_then(|f| f.value_char().ok())
                    .unwrap_or(b'1');

                if let Some(order) = self.orders.get_mut(&resolved_id) {
                    let prev_filled = order.filled_qty();
                    let new_filled_qty = cum_qty;
                    let fill_increment = new_filled_qty - prev_filled;
                    self.total_filled_qty += fill_increment.max(0.0);

                    if ord_status == b'2' {
                        order.state = OrderState::Filled { filled_qty: cum_qty, avg_px };
                        return Ok(OrderEvent::FullFill { qty: last_qty, price: last_px });
                    } else {
                        order.state = OrderState::PartiallyFilled { filled_qty: cum_qty, avg_px };
                    }
                }
                OrderEvent::PartialFill { qty: last_qty, price: last_px }
            }

            ExecType::Fill => {
                let last_qty = msg.get_field(tags::LastQty)
                    .ok_or(FIXError::MissingField(tags::LastQty))?
                    .value_f64()?;
                let last_px = msg.get_field(tags::LastPx)
                    .ok_or(FIXError::MissingField(tags::LastPx))?
                    .value_f64()?;
                let cum_qty = msg.get_field(tags::CumQty)
                    .and_then(|f| f.value_f64().ok())
                    .unwrap_or(last_qty);
                let avg_px = msg.get_field(tags::AvgPx)
                    .and_then(|f| f.value_f64().ok())
                    .unwrap_or(last_px);

                if let Some(order) = self.orders.get_mut(&resolved_id) {
                    let prev_filled = order.filled_qty();
                    self.total_filled_qty += (cum_qty - prev_filled).max(0.0);
                    order.state = OrderState::Filled { filled_qty: cum_qty, avg_px };
                }
                OrderEvent::FullFill { qty: last_qty, price: last_px }
            }

            ExecType::Canceled => {
                if let Some(order) = self.orders.get_mut(&resolved_id) {
                    order.state = OrderState::Cancelled;
                }
                OrderEvent::CancelAck
            }

            ExecType::PendingCancel => {
                if let Some(order) = self.orders.get_mut(&resolved_id) {
                    if !order.state.is_terminal() {
                        order.state = OrderState::PendingCancel;
                    }
                }
                OrderEvent::StateUnchanged
            }

            ExecType::Rejected => {
                let reason = msg.get_field(tags::OrdRejReason)
                    .and_then(|f| f.value_str().ok())
                    .unwrap_or("0")
                    .to_string();
                let text = msg.get_field(tags::Text)
                    .and_then(|f| f.value_str().ok())
                    .unwrap_or("")
                    .to_string();
                let full_reason = if text.is_empty() { reason.clone() } else { format!("{reason}: {text}") };
                if let Some(order) = self.orders.get_mut(&resolved_id) {
                    order.state = OrderState::Rejected { reason: full_reason.clone() };
                }
                OrderEvent::Reject { reason: full_reason }
            }

            _ => OrderEvent::StateUnchanged,
        };

        Ok(event)
    }

    // ---------------------------------------------------------------------------
    // Queries
    // ---------------------------------------------------------------------------

    /// Return all orders that are not in a terminal state.
    pub fn pending_orders(&self) -> Vec<&ManagedOrder> {
        self.orders.values().filter(|o| o.state.is_active()).collect()
    }

    /// Return all orders regardless of state.
    pub fn all_orders(&self) -> Vec<&ManagedOrder> {
        self.orders.values().collect()
    }

    /// Look up an order by ClOrdID.
    pub fn get_order(&self, cl_ord_id: &str) -> Option<&ManagedOrder> {
        self.orders.get(cl_ord_id)
    }

    /// True if the order exists and is in a terminal state.
    pub fn is_order_done(&self, cl_ord_id: &str) -> bool {
        self.orders.get(cl_ord_id).map(|o| o.is_done()).unwrap_or(false)
    }

    /// Ratio of filled qty to submitted qty across all orders.
    /// Returns 0.0 if nothing has been submitted.
    pub fn fill_rate(&self) -> f64 {
        if self.total_submitted_qty < 1e-12 {
            0.0
        } else {
            self.total_filled_qty / self.total_submitted_qty
        }
    }

    /// Number of orders currently tracked.
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }

    /// Count of orders in terminal states.
    pub fn completed_count(&self) -> usize {
        self.orders.values().filter(|o| o.is_done()).count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{FixMessage, tags};

    fn make_exec_report(exec_type: &str, cl_ord_id: &str) -> FixMessage {
        let mut msg = FixMessage::new("FIX.4.2", "8");
        msg.set_field_str(tags::ExecType, exec_type);
        msg.set_field_str(tags::ClOrdID, cl_ord_id);
        msg
    }

    fn add_fill(msg: &mut FixMessage, last_qty: f64, last_px: f64, cum_qty: f64, avg_px: f64, ord_status: &str) {
        msg.set_field_str(tags::LastQty, &last_qty.to_string());
        msg.set_field_str(tags::LastPx, &last_px.to_string());
        msg.set_field_str(tags::CumQty, &cum_qty.to_string());
        msg.set_field_str(tags::AvgPx, &avg_px.to_string());
        msg.set_field_str(tags::OrdStatus, ord_status);
    }

    fn register(mgr: &mut FIXOrderManager, cl_ord_id: &str, qty: f64) {
        mgr.register_order(cl_ord_id, "AAPL", '1', qty, None, 0).unwrap();
    }

    // ---------------------------------------------------------------------------

    #[test]
    fn test_register_order() {
        let mut mgr = FIXOrderManager::new();
        register(&mut mgr, "ORD-001", 100.0);
        assert_eq!(mgr.order_count(), 1);
        let ord = mgr.get_order("ORD-001").unwrap();
        assert_eq!(ord.state, OrderState::PendingNew);
    }

    #[test]
    fn test_duplicate_registration_fails() {
        let mut mgr = FIXOrderManager::new();
        register(&mut mgr, "ORD-001", 100.0);
        let result = mgr.register_order("ORD-001", "AAPL", '1', 100.0, None, 0);
        assert!(matches!(result, Err(FIXError::DuplicateOrder(_))));
    }

    #[test]
    fn test_new_order_ack() {
        let mut mgr = FIXOrderManager::new();
        register(&mut mgr, "ORD-001", 100.0);
        let mut er = make_exec_report("0", "ORD-001");
        er.set_field_str(tags::OrderID, "SRV-001");
        let event = mgr.on_execution_report(&er).unwrap();
        assert_eq!(event, OrderEvent::Accepted);
        assert_eq!(mgr.get_order("ORD-001").unwrap().state, OrderState::New);
    }

    #[test]
    fn test_partial_fill() {
        let mut mgr = FIXOrderManager::new();
        register(&mut mgr, "ORD-002", 200.0);
        let mut er = make_exec_report("1", "ORD-002");
        add_fill(&mut er, 50.0, 100.0, 50.0, 100.0, "1");
        let event = mgr.on_execution_report(&er).unwrap();
        assert!(matches!(event, OrderEvent::PartialFill { qty, .. } if (qty - 50.0).abs() < 1e-9));
        assert!(!mgr.is_order_done("ORD-002"));
    }

    #[test]
    fn test_full_fill() {
        let mut mgr = FIXOrderManager::new();
        register(&mut mgr, "ORD-003", 100.0);
        let mut er = make_exec_report("2", "ORD-003");
        add_fill(&mut er, 100.0, 150.0, 100.0, 150.0, "2");
        let event = mgr.on_execution_report(&er).unwrap();
        assert!(matches!(event, OrderEvent::FullFill { qty, .. } if (qty - 100.0).abs() < 1e-9));
        assert!(mgr.is_order_done("ORD-003"));
    }

    #[test]
    fn test_cancel_ack() {
        let mut mgr = FIXOrderManager::new();
        register(&mut mgr, "ORD-004", 50.0);
        let er = make_exec_report("4", "ORD-004");
        let event = mgr.on_execution_report(&er).unwrap();
        assert_eq!(event, OrderEvent::CancelAck);
        assert_eq!(mgr.get_order("ORD-004").unwrap().state, OrderState::Cancelled);
        assert!(mgr.is_order_done("ORD-004"));
    }

    #[test]
    fn test_reject() {
        let mut mgr = FIXOrderManager::new();
        register(&mut mgr, "ORD-005", 100.0);
        let mut er = make_exec_report("8", "ORD-005");
        er.set_field_str(tags::OrdRejReason, "3");
        er.set_field_str(tags::Text, "price out of range");
        let event = mgr.on_execution_report(&er).unwrap();
        assert!(matches!(event, OrderEvent::Reject { .. }));
        assert!(mgr.is_order_done("ORD-005"));
    }

    #[test]
    fn test_fill_rate_calculation() {
        let mut mgr = FIXOrderManager::new();
        register(&mut mgr, "ORD-006", 100.0);
        register(&mut mgr, "ORD-007", 100.0);

        // Fully fill ORD-006
        let mut er1 = make_exec_report("2", "ORD-006");
        add_fill(&mut er1, 100.0, 50.0, 100.0, 50.0, "2");
        mgr.on_execution_report(&er1).unwrap();

        // Fill rate = 100/200 = 0.5
        assert!((mgr.fill_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_pending_orders_excludes_terminal() {
        let mut mgr = FIXOrderManager::new();
        register(&mut mgr, "ORD-008", 100.0);
        register(&mut mgr, "ORD-009", 100.0);

        let mut er = make_exec_report("4", "ORD-008"); // cancel ORD-008
        mgr.on_execution_report(&er).unwrap();

        er = make_exec_report("0", "ORD-009"); // ack ORD-009 -> still active
        mgr.on_execution_report(&er).unwrap();

        let pending = mgr.pending_orders();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].cl_ord_id, "ORD-009");
    }

    #[test]
    fn test_unknown_order_returns_error() {
        let mut mgr = FIXOrderManager::new();
        let er = make_exec_report("0", "UNKNOWN-999");
        let result = mgr.on_execution_report(&er);
        assert!(matches!(result, Err(FIXError::OrderNotFound(_))));
    }
}
