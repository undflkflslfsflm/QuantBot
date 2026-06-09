from datetime import datetime

inp = "data/VTI.csv"         # your current file
out = "data/VTI_fixed.csv"   # corrected file

with open(inp, "r", encoding="utf-8") as f_in, open(out, "w", encoding="utf-8", newline="\n") as f_out:
    header = f_in.readline().strip()
    # normalize header spacing just in case
    f_out.write("date,close\n")

    for line in f_in:
        line = line.strip()
        if not line:
            continue
        d, close = line.split(",", 1)
        dt = datetime.strptime(d.strip(), "%m/%d/%Y")   # your format
        f_out.write(f"{dt:%Y-%m-%d},{close.strip()}\n")

print("Wrote:", out)
