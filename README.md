# QuantBot

QuantBot is a tactical asset allocation research and IBKR execution tool.

## Live Safety Halt

The `live-ibkr` command refuses to trade when a file named `HALT` exists in the state directory, for example:

```powershell
New-Item -ItemType File .\matvm_state\HALT
```

Delete that file to re-enable live trading:

```powershell
Remove-Item .\matvm_state\HALT
```

