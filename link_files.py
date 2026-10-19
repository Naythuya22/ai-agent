import re

with open('web/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# အရင်က ထည့်ခဲ့တဲ့ ကုဒ်အဟောင်းတွေ ရှိရင် ရှင်းလင်းမယ်
content = re.sub(r'<style>\s*/\* Soft Dark Syntax Highlighting \*/.*?</style>', '', content, flags=re.DOTALL)
content = re.sub(r'<script>\s*document.addEventListener\("DOMContentLoaded".*?</script>', '', content, flags=re.DOTALL)

# CSS ကို <head> ထဲမှာ ထည့်မယ်
if '</head>' in content and 'soft-theme.css' not in content:
    content = content.replace('</head>', '  <link rel="stylesheet" href="soft-theme.css">\n</head>', 1)

# JS ကို </body> အပေါ်မှာ ထည့်မယ်
if '</body>' in content and 'highlighter.js' not in content:
    content = content.replace('</body>', '  <script src="highlighter.js"></script>\n</body>', 1)

with open('web/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ index.html ထဲသို့ CSS နှင့် JS ကို ချိတ်ဆက်ပြီးပါပြီ။")
