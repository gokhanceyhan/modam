

def create_simple_bids_from_hourly_bid(hourly_bid):
    step_id_2_simple_bid = {}
    count = 1
    prev_qnt = 0
    if hourly_bid.price_quantity_pairs[0][1] <= 0:
        # supply bid
        for price, quantity in hourly_bid.price_quantity_pairs:
            step_id_2_simple_bid[count] = (price, quantity - prev_qnt)
            prev_qnt = quantity
            count += 1
    elif hourly_bid.price_quantity_pairs[len(hourly_bid.price_quantity_pairs)-1][1] >= 0:
        # demand bid
        for price, quantity in reversed(hourly_bid.price_quantity_pairs):
            step_id_2_simple_bid[count] = (price, quantity - prev_qnt)
            prev_qnt = quantity
            count += 1
    else:
        prev_qnt = hourly_bid.price_quantity_pairs[0][1]
        prev_prc = hourly_bid.price_quantity_pairs[0][0]
        for price, quantity in hourly_bid.price_quantity_pairs[1:]:
            if quantity >= 0:
                # demand step
                step_id_2_simple_bid[count] = (prev_prc, prev_qnt - quantity)
                prev_prc = price
            elif prev_qnt > 0:
                # first supply step
                step_id_2_simple_bid[count] = (prev_prc, prev_qnt)
                count += 1
                step_id_2_simple_bid[count] = (price, quantity)
            else:
                # supply step
                step_id_2_simple_bid[count] = (price, quantity - prev_qnt)
            prev_qnt = quantity
            count += 1
    return {step_id: simple_bid for step_id, simple_bid in step_id_2_simple_bid.items() if abs(simple_bid[1]) > 0}


def is_accepted_block_bid_pab(bid, mcp):
    mcp = list(mcp.values())
    first = bid.period - 1
    last = bid.period+bid.num_period-1
    mcp_block = mcp[first:last]
    mcp_avg = sum(mcp_block)/len(mcp_block)
    if bid.is_supply:
        if bid.price > mcp_avg:
            return True
        else:
            return False
    else:
        if bid.price < mcp_avg:
            return True
        else:
            return False


def is_rejected_block_bid_prb(bid, mcp):
    mcp = list(mcp.values())
    first = bid.period - 1
    last = bid.period+bid.num_period-1
    mcp_block = mcp[first:last]
    mcp_avg = sum(mcp_block)/len(mcp_block)
    if bid.is_supply:
        if bid.price < mcp_avg:
            return True
        else:
            return False
    else:
        if bid.price > mcp_avg:
            return True
        else:
            return False
