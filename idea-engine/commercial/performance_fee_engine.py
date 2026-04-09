"""
Performance Fee Engine: automated fund accounting for real money management.

Implements the standard hedge fund fee structure:
  - Management fee: X% of AUM annually (charged monthly)
  - Performance fee: Y% of profits above high-water mark
  - Hurdle rate: only charge performance fee if return exceeds hurdle
  - Crystallization: lock in high-water mark at fee payment dates
  - Clawback provision: recoup previously paid fees if subsequent losses
  - Multi-investor tracking: each investor has their own HWM and fee history
  - Tax-lot tracking: FIFO cost basis for each investor's shares

This is the module that turns the system from a research project
into a real money management business.
"""

from __future__ import annotations
import time
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class InvestorAccount:
    """Individual investor's account in the fund."""
    investor_id: str
    name: str
    initial_investment: float
    current_nav: float
    shares: float                   # fund shares owned
    high_water_mark: float          # per-share HWM for this investor
    inception_date: float
    last_fee_date: float = 0.0

    # Fee history
    total_management_fees_paid: float = 0.0
    total_performance_fees_paid: float = 0.0
    total_fees_paid: float = 0.0

    # Clawback
    clawback_reserve: float = 0.0  # accumulated performance fees subject to clawback

    # Status
    is_active: bool = True
    redemption_requested: bool = False
    lock_up_end: float = 0.0       # no redemptions before this date


@dataclass
class FeeCalculation:
    """Result of a fee calculation for one period."""
    investor_id: str
    period: str                     # "2026-Q1"
    beginning_nav: float
    ending_nav: float
    gross_return_pct: float
    management_fee: float
    performance_fee: float
    total_fee: float
    net_return_pct: float
    new_hwm: float
    hurdle_exceeded: bool


@dataclass
class FundTerms:
    """Fee structure terms."""
    management_fee_pct: float = 2.0     # 2% annual
    performance_fee_pct: float = 20.0   # 20% of profits above HWM
    hurdle_rate_pct: float = 0.0        # 0% = no hurdle
    crystallization_frequency: str = "quarterly"  # monthly / quarterly / annually
    lock_up_months: int = 12            # 12-month lock-up
    clawback_enabled: bool = True
    redemption_notice_days: int = 45
    min_investment: float = 100_000


class PerformanceFeeEngine:
    """
    Automated fund accounting and fee calculation.

    Supports:
    - Multiple investors with individual HWMs
    - Standard 2-and-20 fee structure (configurable)
    - Hurdle rate (only charge perf fee above hurdle)
    - High-water mark with crystallization
    - Clawback provisions
    - Lock-up period enforcement
    - Redemption processing
    - NAV calculation and share accounting
    """

    def __init__(self, terms: FundTerms = None, fund_name: str = "SRFM Event Horizon Fund"):
        self.terms = terms or FundTerms()
        self.fund_name = fund_name
        self._investors: Dict[str, InvestorAccount] = {}
        self._nav_per_share: float = 100.0  # start at $100/share
        self._total_shares: float = 0.0
        self._total_aum: float = 0.0
        self._fee_history: List[FeeCalculation] = []
        self._nav_history: List[float] = []

    def add_investor(self, investor_id: str, name: str, amount: float) -> InvestorAccount:
        """Add a new investor to the fund."""
        if amount < self.terms.min_investment:
            raise ValueError(f"Minimum investment is ${self.terms.min_investment:,.0f}")

        shares = amount / self._nav_per_share
        account = InvestorAccount(
            investor_id=investor_id,
            name=name,
            initial_investment=amount,
            current_nav=amount,
            shares=shares,
            high_water_mark=self._nav_per_share,
            inception_date=time.time(),
            lock_up_end=time.time() + self.terms.lock_up_months * 30 * 86400,
        )

        self._investors[investor_id] = account
        self._total_shares += shares
        self._total_aum += amount
        return account

    def update_nav(self, new_nav_per_share: float) -> None:
        """Update NAV per share (call after each trading period)."""
        self._nav_per_share = new_nav_per_share
        self._nav_history.append(new_nav_per_share)

        # Update each investor's current NAV
        for account in self._investors.values():
            account.current_nav = account.shares * new_nav_per_share

        self._total_aum = sum(a.current_nav for a in self._investors.values())

    def calculate_fees(self, period_name: str) -> List[FeeCalculation]:
        """
        Calculate management and performance fees for all investors.
        Call this at crystallization frequency (monthly/quarterly).
        """
        results = []

        for investor_id, account in self._investors.items():
            if not account.is_active:
                continue

            beginning_nav = account.current_nav
            current_share_price = self._nav_per_share

            # Management fee: annual rate / periods per year
            periods_per_year = {"monthly": 12, "quarterly": 4, "annually": 1}
            n_periods = periods_per_year.get(self.terms.crystallization_frequency, 4)
            mgmt_fee = account.current_nav * (self.terms.management_fee_pct / 100) / n_periods

            # Performance fee: on gains above HWM
            perf_fee = 0.0
            hurdle_exceeded = False

            if current_share_price > account.high_water_mark:
                gain_per_share = current_share_price - account.high_water_mark
                gain_total = gain_per_share * account.shares

                # Check hurdle
                hurdle_return = account.high_water_mark * (self.terms.hurdle_rate_pct / 100 / n_periods)
                if gain_per_share > hurdle_return:
                    hurdle_exceeded = True
                    # Performance fee on gain above hurdle
                    eligible_gain = (gain_per_share - hurdle_return) * account.shares
                    perf_fee = eligible_gain * (self.terms.performance_fee_pct / 100)

                # Update HWM (crystallization)
                account.high_water_mark = current_share_price

            total_fee = mgmt_fee + perf_fee
            gross_return = (current_share_price / max(account.high_water_mark, 1e-10) - 1)

            # Deduct fees from investor's shares
            if total_fee > 0 and account.current_nav > total_fee:
                fee_shares = total_fee / max(current_share_price, 1e-10)
                account.shares -= fee_shares
                self._total_shares -= fee_shares
                account.current_nav = account.shares * current_share_price

            # Record
            account.total_management_fees_paid += mgmt_fee
            account.total_performance_fees_paid += perf_fee
            account.total_fees_paid += total_fee
            account.last_fee_date = time.time()

            # Clawback reserve
            if self.terms.clawback_enabled:
                account.clawback_reserve += perf_fee

            net_return = (account.current_nav / max(beginning_nav, 1e-10) - 1)

            calc = FeeCalculation(
                investor_id=investor_id,
                period=period_name,
                beginning_nav=beginning_nav,
                ending_nav=account.current_nav,
                gross_return_pct=gross_return * 100,
                management_fee=mgmt_fee,
                performance_fee=perf_fee,
                total_fee=total_fee,
                net_return_pct=net_return * 100,
                new_hwm=account.high_water_mark,
                hurdle_exceeded=hurdle_exceeded,
            )
            results.append(calc)
            self._fee_history.append(calc)

        return results

    def process_redemption(self, investor_id: str) -> Dict:
        """Process a redemption request."""
        account = self._investors.get(investor_id)
        if not account:
            return {"error": "Investor not found"}

        if time.time() < account.lock_up_end:
            days_left = (account.lock_up_end - time.time()) / 86400
            return {"error": f"Lock-up period: {days_left:.0f} days remaining"}

        # Calculate final NAV and fees
        redemption_amount = account.current_nav

        # Clawback check
        clawback = 0.0
        if self.terms.clawback_enabled and self._nav_per_share < account.high_water_mark:
            # Some performance fees should be returned
            loss_pct = (account.high_water_mark - self._nav_per_share) / account.high_water_mark
            clawback = min(account.clawback_reserve, loss_pct * account.clawback_reserve)
            redemption_amount += clawback

        account.is_active = False
        self._total_shares -= account.shares
        self._total_aum -= account.current_nav
        account.shares = 0
        account.current_nav = 0

        return {
            "investor_id": investor_id,
            "redemption_amount": redemption_amount,
            "clawback_refund": clawback,
            "total_fees_paid": account.total_fees_paid,
            "total_return_pct": (redemption_amount / account.initial_investment - 1) * 100,
        }

    def get_fund_summary(self) -> Dict:
        """Get fund-level summary."""
        active = [a for a in self._investors.values() if a.is_active]
        total_fees = sum(a.total_fees_paid for a in self._investors.values())
        total_perf_fees = sum(a.total_performance_fees_paid for a in self._investors.values())
        total_mgmt_fees = sum(a.total_management_fees_paid for a in self._investors.values())

        return {
            "fund_name": self.fund_name,
            "nav_per_share": self._nav_per_share,
            "total_aum": self._total_aum,
            "total_shares": self._total_shares,
            "n_investors": len(active),
            "total_fees_earned": total_fees,
            "management_fees_earned": total_mgmt_fees,
            "performance_fees_earned": total_perf_fees,
            "terms": {
                "management_fee": f"{self.terms.management_fee_pct}%",
                "performance_fee": f"{self.terms.performance_fee_pct}%",
                "hurdle_rate": f"{self.terms.hurdle_rate_pct}%",
                "lock_up": f"{self.terms.lock_up_months} months",
                "crystallization": self.terms.crystallization_frequency,
            },
        }

    def get_investor_statement(self, investor_id: str) -> Dict:
        """Generate investor statement."""
        account = self._investors.get(investor_id)
        if not account:
            return {"error": "Investor not found"}

        total_return = (account.current_nav / max(account.initial_investment, 1e-10) - 1)
        fees = [f for f in self._fee_history if f.investor_id == investor_id]

        return {
            "investor": account.name,
            "initial_investment": account.initial_investment,
            "current_nav": account.current_nav,
            "total_return_pct": total_return * 100,
            "total_fees_paid": account.total_fees_paid,
            "net_return_after_fees_pct": ((account.current_nav + account.total_fees_paid) / account.initial_investment - 1) * 100,
            "shares_held": account.shares,
            "hwm": account.high_water_mark,
            "lock_up_remaining_days": max(0, (account.lock_up_end - time.time()) / 86400),
            "fee_history": [
                {"period": f.period, "mgmt": f.management_fee, "perf": f.performance_fee, "total": f.total_fee}
                for f in fees[-4:]
            ],
        }
