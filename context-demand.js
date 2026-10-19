// SPDX-License-Identifier: MIT
// All charts and tables are server-rendered; these controls are optional.
const viewButtons = Array.from(document.querySelectorAll('[data-view]'))
const views = Array.from(document.querySelectorAll('[data-trend-view]'))
if (viewButtons.length && views.length) {
    document.querySelector('.study-controls').hidden = false
    const chooseView = (name) => {
        viewButtons.forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.view === name)))
        views.forEach((panel) => {
            panel.hidden = panel.dataset.trendView !== name
        })
        document.querySelector('[data-series-legend="text"]').hidden = name === 'ratio'
    }
    viewButtons.forEach((button) => button.addEventListener('click', () => chooseView(button.dataset.view)))
    chooseView('ratio')
}

const architectureSelect = document.querySelector('[data-architecture-select]')
if (architectureSelect) {
    architectureSelect.parentElement.hidden = false
    const chooseArchitecture = () => {
        document.querySelectorAll('[data-architecture-panel]').forEach((panel) => {
            panel.hidden = panel.dataset.architecturePanel !== architectureSelect.value
        })
    }
    architectureSelect.addEventListener('change', chooseArchitecture)
    chooseArchitecture()
}

const tooltip = document.querySelector('.chart-tooltip')
if (tooltip) {
    document.querySelectorAll('[data-chart-point]').forEach((point) => {
        const show = () => {
            tooltip.textContent = point.dataset.chartPoint
            tooltip.hidden = false
        }
        const hide = () => {
            tooltip.hidden = true
        }
        point.addEventListener('pointerenter', show)
        point.addEventListener('pointerleave', hide)
        point.addEventListener('focus', show)
        point.addEventListener('blur', hide)
    })
}
