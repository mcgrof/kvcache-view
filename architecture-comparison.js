// SPDX-License-Identifier: MIT
// The comparison remains readable, with all datasets and measures, without JS.
const comparisonSelect = document.querySelector('[data-comparison-select]')
const comparisonButtons = Array.from(document.querySelectorAll('[data-comparison-mode]'))
let comparisonMode = 'share'

if (comparisonSelect && comparisonButtons.length) {
    document.querySelector('.architecture-market-controls').hidden = false
    const updateComparison = () => {
        document.querySelectorAll('[data-comparison-dataset]').forEach((panel) => {
            panel.hidden = panel.dataset.comparisonDataset !== comparisonSelect.value
            panel.querySelectorAll('[data-comparison-view]').forEach((view) => {
                view.hidden = view.dataset.comparisonView !== comparisonMode
            })
        })
        comparisonButtons.forEach((button) => {
            button.setAttribute('aria-pressed', String(button.dataset.comparisonMode === comparisonMode))
        })
        const tooltip = document.querySelector('.chart-tooltip')
        if (tooltip) tooltip.hidden = true
    }
    comparisonSelect.addEventListener('change', updateComparison)
    comparisonButtons.forEach((button) => {
        button.addEventListener('click', () => {
            comparisonMode = button.dataset.comparisonMode
            updateComparison()
        })
    })
    updateComparison()
}

const comparisonTooltip = document.querySelector('.chart-tooltip')
if (comparisonTooltip) {
    document.querySelectorAll('[data-chart-point]').forEach((point) => {
        const show = () => {
            comparisonTooltip.textContent = point.dataset.chartPoint
            comparisonTooltip.hidden = false
        }
        const hide = () => {
            comparisonTooltip.hidden = true
        }
        point.addEventListener('pointerenter', show)
        point.addEventListener('pointerleave', hide)
        point.addEventListener('focus', show)
        point.addEventListener('blur', hide)
    })
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') comparisonTooltip.hidden = true
    })
}
