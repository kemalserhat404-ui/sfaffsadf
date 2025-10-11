def calculate_position_size(account_balance, risk_per_trade, stop_loss_pips):
    """
    account_balance: float, toplam bakiye
    risk_per_trade: float, 0.01 = %1 risk
    stop_loss_pips: float, pip farkı
    """
    return (account_balance * risk_per_trade) / stop_loss_pips
