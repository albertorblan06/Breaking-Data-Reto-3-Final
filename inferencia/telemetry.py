from multi_agent_ale_py import ALEInterface
ale = ALEInterface()
ale.loadROM('.venv/lib/python3.9/site-packages/AutoROM/roms/boxing.bin')
meanings = ale.getLegalActionSet()
print(meanings)
for a in meanings:
    print(f"{a}")
