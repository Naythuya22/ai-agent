document.addEventListener("DOMContentLoaded", function() {
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === 1) {
          if (node.tagName === 'PRE') highlightBlock(node);
          else if (node.querySelectorAll) {
             node.querySelectorAll('pre').forEach(highlightBlock);
          }
        }
      });
    });
  });
  const chatContainer = document.getElementById('chat') || document.body;
  observer.observe(chatContainer, { childList: true, subtree: true });
  document.querySelectorAll('pre code').forEach((block) => highlightBlock(block.parentElement));

  function highlightBlock(pre) {
    if (!pre || pre.dataset.highlighted) return;
    const code = pre.querySelector('code');
    if (!code) return;

    let text = code.textContent || "";
    let html = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    let placeholders = [];
    function save(match, type) {
        placeholders.push('<span class="token ' + type + '">' + match + '</span>');
        return '___TOKEN_' + (placeholders.length - 1) + '___';
    }

    html = html.replace(/(\/\/[^\n]*|\/\*[\s\S]*?\*\/|#[^\n]*)/g, (m) => save(m, 'comment'));
    // Python ၏ သုံးထပ်ကွက်စာများကို ရှာရန် (Error မတက်စေရန် \x22 ကို သုံးထားသည်)
    html = html.replace(/(\x22\x22\x22[\s\S]*?\x22\x22\x22|\x27\x27\x27[\s\S]*?\x27\x27\x27)/g, (m) => save(m, 'string'));
    html = html.replace(/(['"`])(.*?)\1/g, (m) => save(m, 'string'));
    html = html.replace(/\b(function|const|let|var|if|else|return|for|while|class|import|export|def|from|as|async|await|try|catch|except|pass|raise|with|yield|lambda|true|false|null|None|True|False)\b/g, (m) => save(m, 'keyword'));
    html = html.replace(/\b(\d+\.?\d*)\b/g, (m) => save(m, 'number'));
    html = html.replace(/\b([a-zA-Z_]\w*)\s*(?=\()/g, (m) => save(m, 'function'));

    html = html.replace(/___TOKEN_(\d+)___/g, (m, i) => placeholders[i]);
    code.innerHTML = html;
    pre.dataset.highlighted = "true";
  }
});
