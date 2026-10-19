import re

file_path = 'web/index.html'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# CSS အသစ်ကို အတင်းအကြပ် ထည့်သွင်းခြင်း (ဘယ်အရောင်ကမှ မလွှမ်းမိုးနိုင်အောင် !important သုံးထားသည်)
force_css = """
<style id="force-tool-colors">
  :root {
    --tool-bg: #1e2233 !important;
    --tool-border: #3b4266 !important;
    --tool-text: #a9b1d6 !important;
  }
  .msg.tool {
    background: #1e2233 !important;
    border-left: 3px solid #3b4266 !important;
    color: #a9b1d6 !important;
  }
</style>
"""

# အရင်က ထည့်ခဲ့တဲ့ force-tool-colors ရှိရင် အရင်ဖျက်မယ်
content = re.sub(r'<style id="force-tool-colors">.*?</style>', '', content, flags=re.DOTALL)

# </head> ရဲ့ အပေါ်မှာ ထည့်မယ်
if '</head>' in content:
    content = content.replace('</head>', f'{force_css}\n</head>', 1)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Tool အရောင်များကို Slate Blue (အပြာနွယ်ရောင်) သို့ အတင်းအကြပ် ပြောင်းလဲပြီးပါပြီ။")
else:
    print("❌ </head> tag ကို ရှာမတွေ့ပါ။")
