with open('app.py', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# Fix line 1177 (0-indexed = line 1178 displayed) - add indentation
if len(lines) > 1177:
    lines[1177] = '    ' + lines[1177].lstrip()

with open('app.py', 'w', encoding='utf-8', errors='ignore') as f:
    f.writelines(lines)

print("Indentation fixed!")
