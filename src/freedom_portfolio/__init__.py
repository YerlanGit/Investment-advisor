"""
freedom_portfolio — read-only Freedom Broker (Tradernet) portfolio connector.

Public surface:
    auth.build_signature, auth.build_request
    client.TradernetClient, client.AuthenticationError, client.InvalidSignatureError
    models.Portfolio, models.Position, models.AccountBalance
    websocket.TradernetWebSocket          (optional — requires python-socketio)
    display.render_portfolio_table        (optional — requires rich)
"""

from freedom_portfolio.auth import build_request, build_signature
from freedom_portfolio.client import (
    AuthenticationError,
    BrokerAPIError,
    EmptyPortfolioError,
    InvalidSignatureError,
    TradernetClient,
)
from freedom_portfolio.models import AccountBalance, Portfolio, Position

__all__ = [
    "AccountBalance",
    "AuthenticationError",
    "BrokerAPIError",
    "EmptyPortfolioError",
    "InvalidSignatureError",
    "Portfolio",
    "Position",
    "TradernetClient",
    "build_request",
    "build_signature",
]
