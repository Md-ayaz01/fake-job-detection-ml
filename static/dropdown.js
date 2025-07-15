document.querySelectorAll('.custom-dropdown').forEach(drop => {
    const selected = drop.querySelector('.selected');
    const list = drop.querySelector('.dropdown-list');
    const hiddenInput = drop.querySelector('input[type="hidden"]');

    selected.addEventListener('click', () => {
        list.style.display = list.style.display === 'block' ? 'none' : 'block';
    });

    list.querySelectorAll('li').forEach(item => {
        item.addEventListener('click', () => {
            selected.innerText = item.innerText;
            hiddenInput.value = item.getAttribute('data-value');
            list.style.display = 'none';
        });
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!drop.contains(e.target)) {
            list.style.display = 'none';
        }
    });
});
