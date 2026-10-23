# Beta စမ်းသပ်သူ လမ်းညွှန် (မြန်မာ)

ဒီ app က Android ဖုန်း (Termux) ပေါ်မှာ run တဲ့ AI assistant ပါ။
အခုက **စမ်းသပ်ဆင့် (beta)** ဖြစ်လို့ အမှား ရှိနိုင်ပါတယ် — တွေ့ရင်
အောက်မှာ ရေးထားတဲ့ နည်းနဲ့ feedback ပို့ပေးပါ။ ကိုယ့် version ကို
app ရဲ့ **Settings → System → Version** မှာ ကြည့်နိုင်ပါတယ်။

## ၁။ Termux သွင်းရန်

- **F-Droid** (f-droid.org) ဒါမှမဟုတ် **GitHub** (github.com/termux/termux-app/releases)
  ကနေ သွင်းပါ။
- ⚠️ **Play Store က Termux က ခေတ်ကုန်နေပါတယ်** — အဲဒီက မသွင်းပါနဲ့၊
  install အဆင့်မှာ ပိတ်မိနေတတ်ပါတယ်။

Termux ဖွင့်ပြီး ပထမဆုံး တစ်ခါ:

```
pkg update -y
termux-setup-storage
```

(storage ခွင့်ပြုချက် တောင်းရင် Allow နှိပ်ပါ — ဖိုင် တွေ ဖတ်/ရေးဖို့ လိုပါတယ်။)

## ၂။ Project သွင်းရန်

```
pkg install -y git
git clone <repo လိပ်စာ> ai-agent
cd ai-agent
sh install.sh
```

`Installed: ai-agent v…` လို့ နောက်ဆုံး ပေါ်ရင် သွင်းပြီး ပါပြီ။
(Code mode ပါ သုံးချင်ရင် `sh install.sh --with-code` လို့ run ပါ။)

## ၃။ API key ရယူရန် (အခမဲ့)

Key မရှိရင် app က ဘာမှ ဖြေမှာ မဟုတ်ပါဘူး။ အလွယ်ဆုံးက Gemini key ပါ:

1. ဖုန်း browser မှာ **aistudio.google.com** ကို သွားပါ
2. ကိုယ့် **Google အကောင့်** နဲ့ ဝင်ပါ
3. **Get API key → Create API key** နှိပ်ပြီး key ကို copy ယူထားပါ

နောက်ထပ် ရွေးစရာ: **openrouter.ai** မှာ အကောင့် ဖွင့်ပြီး key ထုတ်ရင်
free model တွေ သုံးနိုင်ပါတယ်။

⚠️ ကိုယ့် key ကို ကိုယ်ပဲ သုံးပါ — ဘယ်သူ့ကိုမှ မပေးပါနဲ့။

## ၄။ Run ရန် + key ထည့်ရန်

```
cd ai-agent
python3 server.py
```

ပေါ်လာတဲ့ **token ပါတဲ့ link** ကို browser မှာ ဖွင့်ပါ။ ပြီးရင်:

1. **Settings** ကို ဖွင့်ပါ
2. Gemini provider မှာ copy ထားတဲ့ key ကို ထည့်ပြီး သိမ်းပါ
3. စာပို့ပြီး စမ်းကြည့်ပါ — အဖြေ ပြန်လာရင် အဆင်ပြေပါပြီ

**App လို သုံးချင်ရင်:** browser ရဲ့ menu (⋮) → **Add to Home screen**
နှိပ်လိုက်ရင် home screen မှာ icon ပေါ်လာပြီး app လို ဖွင့်လို့ ရပါတယ်။

## ၅။ Update လုပ်ရန်

Version အသစ် ထွက်ရင်:

```
cd ai-agent
sh update.sh
```

`Updated: v… -> v…` လို့ ပေါ်ရင် server ကို ပိတ်ပြီး `python3 server.py`
နဲ့ ပြန်ဖွင့်ပါ။ (ကိုယ့် ပြင်ထားချက် တွေ ရှိရင် update ရပ်ပြီး
အကြောင်းကြားပါလိမ့်မယ် — ဘာမှ မပျက်ပါဘူး။)

## ၆။ သိထားသင့်တဲ့ ကန့်သတ်ချက်များ

- **429 / Limit ပြည့်သွားပြီ** — free key ရဲ့ တစ်ရက်တာ ပမာဏ ကုန်သွားတာပါ။
  ခဏ စောင့် (ဒါမှမဟုတ် နောက်ရက်) ပြန် စမ်းပါ။ အမှား မဟုတ်ပါဘူး။
- **402 / credit မရှိဘူး** — အဲဒီ provider အကောင့်မှာ ပိုက်ဆံ မရှိတာပါ။
  Free model ပြောင်းသုံး (ဒါမှဟုတ် Gemini key သုံး) ပါ။
- **Local model မထည့်ထားရင် internet လိုပါတယ်။** Model ဖိုင်
  (.litertlm) ထည့်နည်း ကို QUICKSTART.my.md မှာ ကြည့်ပါ။
- Beta ဖြစ်လို့ တစ်ခါတလေ အဖြေ နှေး/ရပ် သွားနိုင်ပါတယ် — feedback ပို့ပေးပါ။

## ၇။ Feedback ပို့နည်း

ပြဿနာ တွေ့ရင် ဒီအတိုင်း ပို့ပေးပါ:

1. App ထဲက **Settings → System → Run checks (Doctor)** ကို နှိပ်ပါ
2. ထွက်လာတဲ့ စာသား (log) ကို copy ယူပါ — key တွေ ဖျက်ပြီး သား
   (redacted) ဖြစ်လို့ ပို့လို့ စိတ်ချရပါတယ်
3. အောက် မေးခွန်း ၃ ခု ကို ဖြေပြီး log နဲ့ အတူ ပို့ပေးပါ:
   - ဘာ လုပ်ဖို့ ကြိုးစားတာလဲ
   - ဘာ ဖြစ်သွားလဲ (ပေါ်တဲ့ စာသား အတိုင်း)
   - ဘာ ဖြစ်စေချင်တာလဲ
4. ကိုယ့် **version** (Settings → System → Version) ကိုပါ ထည့်ပြောပေးပါ
