import sys

for line in sys.stdin:
    trades = [t.split(",") for t in line.strip().split(";")]
    wma = {}
    total_quantity = {}
    last_sequence = {}

    for trade in trades:
        key, value, quantity, sequence = trade[0], float(trade[1]), int(trade[2]), int(trade[3])

        if key not in last_sequence or sequence > last_sequence[key]:
            if key not in wma:
                wma[key] = value
                total_quantity[key] = quantity
            else:
                wma[key] = (wma[key] * total_quantity[key] + value * quantity) / (total_quantity[key] + quantity)
                total_quantity[key] += quantity
            
            last_sequence[key] = sequence
            print(f"{key}: {wma[key]:.2f}")