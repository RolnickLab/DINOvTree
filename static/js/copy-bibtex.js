document.addEventListener('DOMContentLoaded', function() {
    const copyBtn = document.getElementById('copy-bibtex-btn');
    
    // Safety check to ensure the button exists on the current page
    if (copyBtn) {
        copyBtn.addEventListener('click', function() {
            const bibtexText = document.getElementById('bibtex-content').innerText;
            
            navigator.clipboard.writeText(bibtexText).then(function() {
                const originalHTML = copyBtn.innerHTML;
                copyBtn.innerHTML = '<span class="icon is-small"><i class="fas fa-check"></i></span><span>Copied!</span>';
                
                setTimeout(function() {
                    copyBtn.innerHTML = originalHTML;
                }, 2000);
            }).catch(function(err) {
                console.error('Could not copy text: ', err);
            });
        });
    }
});