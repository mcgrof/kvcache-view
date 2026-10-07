// SPDX-License-Identifier: MIT
// Progressive enhancement only: all chart and model data is rendered as HTML.
document.querySelectorAll('[data-model-table]').forEach((table) => {
    const section = table.closest('section')
    const search = section.querySelector('[data-model-search]')
    const category = section.querySelector('[data-category-filter]')
    const status = section.querySelector('[data-result-count]')
    const rows = Array.from(table.tBodies[0].rows)
    const haystacks = rows.map((row) => row.textContent.toLowerCase())
    const filter = () => {
        const query = search.value.trim().toLowerCase()
        const selected = category.value
        let count = 0
        rows.forEach((row, index) => {
            const match =
                (!selected || row.dataset.category === selected) && (!query || haystacks[index].includes(query))
            row.hidden = !match
            if (match) count++
        })
        status.textContent = `${count} of ${rows.length} model variants shown.`
    }
    search.addEventListener('input', filter)
    category.addEventListener('change', filter)
})
