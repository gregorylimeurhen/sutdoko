"use strict"

const ui = {
	input: document.querySelector("#input input"),
	output: document.getElementById("output"),
}

let frame = 0
let busy = false
let ready = false
let queuedText = ""
let worker = null


function render(rows) {
	ui.output.replaceChildren()
	for (const row of rows) {
		const item = document.createElement("div")
		const name = document.createElement("h2")
		const address = document.createElement("p")
		name.textContent = row[0]
		address.textContent = row[1]
		item.append(name, address)
		ui.output.append(item)
	}
}


function flush() {
	if (!worker || !ready || busy) {
		return
	}
	const text = queuedText
	queuedText = null
	busy = true
	worker.postMessage({text, type: "solve"})
}


function update() {
	if (!ready) {
		render([])
		return
	}
	queuedText = ui.input.value
	if (!queuedText) {
		render([])
	}
	flush()
}


function queueUpdate() {
	if (frame) {
		return
	}
	frame = requestAnimationFrame(() => {
		frame = 0
		update()
	})
}


function boot() {
	worker = new Worker("./worker.js")
	worker.addEventListener("message", ({data}) => {
		if (!data || typeof data !== "object") {
			return
		}
		if (data.type === "ready") {
			ready = true
			ui.input.disabled = false
			ui.input.focus()
			update()
			return
		}
		if (data.type === "result") {
			busy = false
			if (queuedText == null && data.text === ui.input.value) {
				render(data.rows || [])
			}
			if (queuedText == null && data.text !== ui.input.value) {
				queuedText = ui.input.value
			}
			flush()
			return
		}
		if (data.type === "error") {
			busy = false
			render([])
			if (queuedText == null) {
				queuedText = ui.input.value
			}
			flush()
		}
	})
	worker.addEventListener("error", () => {
		busy = false
		render([])
	})
}


ui.input.addEventListener("input", queueUpdate)


boot()
