# အမြန် စတင်ရန် လမ်းညွှန် (မြန်မာ)

ဒီ app က Android ဖုန်း (Termux) ပေါ်မှာ run တဲ့ AI assistant ပါ။
ဖုန်းထဲက local model (Gemma) နဲ့ အင်တာနက် မရှိလည်း အလုပ် လုပ်နိုင်ပြီး
key ထည့်လိုက်ရင် cloud AI တွေ (OpenAI, Gemini, Anthropic, OpenRouter,
DeepSeek) ကိုပါ သုံးနိုင်ပါတယ်။

## ၁။ သွင်းရန်

Termux မှာ — အလွယ်ဆုံးက installer ကို run လိုက်တာပါ။

```
pkg install git
git clone <ဒီ repo ရဲ့ လိပ်စာ> ai-agent
cd ai-agent
sh install.sh
```

Code mode ပါ သုံးမယ်ဆို `sh install.sh --with-code` လို့ run ပါ။
လက်နဲ့ တစ်ခုချင်း သွင်းချင်ရင် —

```
pkg install python python-pip git
pip install -r requirements.txt
```

## ၂။ Local model ထားရန် (သုံးချင်မှ — မဖြစ်မနေ မလိုပါ)

`.litertlm` ဖိုင်ကို အောက်ပါ နေရာ တစ်ခုခုမှာ ထားပါ — app က သူ့အလိုလို ရှာတွေ့ပါတယ်။

- project folder ထဲက `models/`
- `Download/gemmamodel/`

Model မရှိလည်း server တက်ပါတယ် — UI, Settings နဲ့ cloud အလှည့်တွေ
အလုပ် လုပ်ပြီး local အလှည့်တွေမှာသာ “Model not found” လို့ ပြန်ပါမယ်။

## ၃။ Run ရန်

```
python server.py
```

ပွင့်လာတဲ့ screen မှာ Server လိပ်စာ (ပုံမှန် `http://127.0.0.1:8080`) နဲ့
token ပါတဲ့ link ပြပါမယ်။ အဲဒီ link ကို ဖုန်း browser မှာ ဖွင့်ပါ။

Terminal ကနေ စကားပြောချင်ရင် (server ပွင့်နေတုန်း) —

```
python app.py
```

## ၄။ Cloud key ထည့်ရန် (သုံးချင်မှ)

Web UI ရဲ့ **Settings** မှာ provider တစ်ခုချင်းစီအတွက် key ထည့်ပါ။
Key တွေကို Termux ရဲ့ private နေရာ (app data) ထဲမှာပဲ သိမ်းပါတယ် —
project folder ထဲမှာ မရှိပါဘူး။ Backup zip ထဲမှာ key တွေ ပါတာမို့
အဲဒီ ဖိုင်ကို ဘယ်သူ့ကိုမှ မပေးပါနဲ့။

## ၅။ Code mode (သုံးချင်မှ)

```
pkg install nodejs-lts
npm install -g opencode-ai
```

## ၆။ Backup လုပ်ရန် (အရေးကြီးပါတယ်)

Memory, session နဲ့ key တွေက ဖုန်း ပျက်ရင် ပြန် ရဖို့ မလွယ်ပါဘူး။
ပုံမှန် backup လုပ်ထားပါ —

```
python backup.py export              # key အပါအဝင် (လျှို့ဝှက် သိမ်းပါ)
python backup.py export --no-secrets # key မပါတဲ့ မူ
python backup.py list <backup.zip>   # ပါဝင်တာ ကြည့်ရန်
python backup.py import <backup.zip> # ပြန် သွင်းရန် (လက်ရှိ data ကို အရင် snapshot ယူပါတယ်)
```

## အကူအညီ

- အသေးစိတ် စာရွက်: `README.md`
- ပြဿနာ တွေ့ရင်: `~/.ai-agent/logs/agent.log` မှာ မှတ်တမ်း ရှိပါတယ်
  (app ထဲမှာ `/logs` လို့လည်း ကြည့်နိုင်ပါတယ်)။
