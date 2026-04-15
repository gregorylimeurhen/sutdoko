"use strict"

const EPS = 1.1920928955078125e-7

const ui = {
	form: document.getElementById("form"),
	input: document.getElementById("input"),
	out: document.getElementById("out"),
	run: document.getElementById("run"),
	status: document.getElementById("status"),
}


class Rng {
	constructor(seed) {
		let x = seed >>> 0
		x ||= 1
		this.s = x
	}

	next() {
		let x = this.s >>> 0
		x ^= (x << 13) >>> 0
		x ^= x >>> 17
		x ^= (x << 5) >>> 0
		x >>>= 0
		this.s = x
		return x / 4294967296
	}

	int(n) {
		return Math.floor(this.next() * n)
	}
}


class Model {
	constructor(assets, buf) {
		this.cfg = assets.config
		this.rooms = assets.rooms
		this.roomMap = assets.room_lookup
		this.roomSet = new Set(this.rooms)
		this.seed = assets.seed
		this.tok = assets.tokenizer
		this.trie = assets.trie
		this.vocab = this.tok.vocab
		this.stoi = {}
		this.nEmbd = this.cfg.n_embd
		this.nHead = this.cfg.n_head
		this.headDim = this.nEmbd / this.nHead
		this.scale = 1 / Math.sqrt(this.headDim)
		this.ws = {}
		for (let i = 0; i < this.vocab.length; i += 1) {
			this.stoi[this.vocab[i]] = i
		}
		for (const name of Object.keys(assets.tensors)) {
			const info = assets.tensors[name]
			this.ws[name] = new Float32Array(buf, info.offset, info.size)
		}
		this.rot = this.buildRotary()
	}

	buildRotary() {
		const len = this.cfg.sequence_len
		const half = this.headDim / 2
		const cos = new Float32Array(len * half)
		const sin = new Float32Array(len * half)
		for (let step = 0; step < len; step += 1) {
			for (let i = 0; i < half; i += 1) {
				const rate = 10000 ** (-(i * 2) / this.headDim)
				const angle = step * rate
				const idx = step * half + i
				cos[idx] = Math.cos(angle)
				sin[idx] = Math.sin(angle)
			}
		}
		return {cos, half, sin}
	}

	normalize(text) {
		return text.trim().toLowerCase()
	}

	encodeText(text) {
		const ids = []
		const unk = this.tok.unk_id
		for (const ch of text) {
			ids.push(this.stoi[ch] ?? unk)
		}
		return ids
	}

	decodeText(ids) {
		let text = ""
		const unk = this.tok.unk_id
		for (const id of ids) {
			if (id > unk) {
				text += this.vocab[id]
			}
		}
		return text
	}

	addInto(dst, src) {
		for (let i = 0; i < dst.length; i += 1) {
			dst[i] += src[i]
		}
	}

	embed(ids) {
		const seq = ids.length
		const dim = this.nEmbd
		const out = new Float32Array(seq * dim)
		const w = this.ws["wte.weight"]
		for (let s = 0; s < seq; s += 1) {
			const row = ids[s] * dim
			const off = s * dim
			out.set(w.subarray(row, row + dim), off)
		}
		return out
	}

	rmsSeq(x, seq, dim, weight) {
		const out = new Float32Array(x.length)
		for (let s = 0; s < seq; s += 1) {
			const off = s * dim
			let sum = 0
			for (let i = 0; i < dim; i += 1) {
				const v = x[off + i]
				sum += v * v
			}
			const scale = 1 / Math.sqrt(sum / dim + EPS)
			for (let i = 0; i < dim; i += 1) {
				let v = x[off + i] * scale
				if (weight) {
					v *= weight[i]
				}
				out[off + i] = v
			}
		}
		return out
	}

	rmsVec(x, dim, weight) {
		return this.rmsSeq(x, 1, dim, weight)
	}

	linearSeq(x, seq, outDim, inDim, weight) {
		const out = new Float32Array(seq * outDim)
		for (let s = 0; s < seq; s += 1) {
			const xOff = s * inDim
			const yOff = s * outDim
			for (let o = 0; o < outDim; o += 1) {
				let sum = 0
				const wOff = o * inDim
				for (let i = 0; i < inDim; i += 1) {
					sum += weight[wOff + i] * x[xOff + i]
				}
				out[yOff + o] = sum
			}
		}
		return out
	}

	linearVec(x, outDim, inDim, weight) {
		return this.linearSeq(x, 1, outDim, inDim, weight)
	}

	splitHeads(x, seq) {
		const heads = []
		for (let h = 0; h < this.nHead; h += 1) {
			heads.push(new Float32Array(seq * this.headDim))
		}
		for (let s = 0; s < seq; s += 1) {
			const row = s * this.nEmbd
			for (let h = 0; h < this.nHead; h += 1) {
				const a = row + h * this.headDim
				const b = s * this.headDim
				const c = a + this.headDim
				heads[h].set(x.subarray(a, c), b)
			}
		}
		return heads
	}

	combineHeads(heads, seq) {
		const out = new Float32Array(seq * this.nEmbd)
		for (let s = 0; s < seq; s += 1) {
			const row = s * this.nEmbd
			for (let h = 0; h < this.nHead; h += 1) {
				const a = s * this.headDim
				const b = row + h * this.headDim
				const c = a + this.headDim
				out.set(heads[h].subarray(a, c), b)
			}
		}
		return out
	}

	applyRotary(heads, seq, start) {
		const cos = this.rot.cos
		const sin = this.rot.sin
		const half = this.rot.half
		for (const head of heads) {
			for (let s = 0; s < seq; s += 1) {
				const rot = (start + s) * half
				const off = s * this.headDim
				for (let i = 0; i < half; i += 1) {
					const a = off + i * 2
					const b = a + 1
					const c = cos[rot + i]
					const d = sin[rot + i]
					const left = head[a]
					const right = head[b]
					head[a] = left * c - right * d
					head[b] = left * d + right * c
				}
			}
		}
	}

	normHeads(heads, seq) {
		const out = []
		for (const head of heads) {
			out.push(this.rmsSeq(head, seq, this.headDim))
		}
		return out
	}

	attendSeq(qh, kh, vh, seq) {
		const out = []
		for (let h = 0; h < this.nHead; h += 1) {
			const q = qh[h]
			const k = kh[h]
			const v = vh[h]
			const y = new Float32Array(seq * this.headDim)
			for (let t = 0; t < seq; t += 1) {
				const tOff = t * this.headDim
				const scores = new Float32Array(t + 1)
				let best = -Infinity
				for (let j = 0; j <= t; j += 1) {
					const jOff = j * this.headDim
					let dot = 0
					for (let i = 0; i < this.headDim; i += 1) {
						dot += q[tOff + i] * k[jOff + i]
					}
					dot *= this.scale
					scores[j] = dot
					if (dot > best) {
						best = dot
					}
				}
				let total = 0
				for (let j = 0; j <= t; j += 1) {
					const w = Math.exp(scores[j] - best)
					scores[j] = w
					total += w
				}
				for (let i = 0; i < this.headDim; i += 1) {
					let sum = 0
					for (let j = 0; j <= t; j += 1) {
						const jOff = j * this.headDim
						sum += scores[j] * v[jOff + i]
					}
					y[tOff + i] = sum / total
				}
			}
			out.push(y)
		}
		return this.combineHeads(out, seq)
	}

	appendCache(cache, kh, vh) {
		const size = cache.size + 1
		const ks = []
		const vs = []
		for (let h = 0; h < this.nHead; h += 1) {
			const nextK = new Float32Array(size * this.headDim)
			const nextV = new Float32Array(size * this.headDim)
			nextK.set(cache.k[h], 0)
			nextV.set(cache.v[h], 0)
			nextK.set(kh[h], cache.size * this.headDim)
			nextV.set(vh[h], cache.size * this.headDim)
			ks.push(nextK)
			vs.push(nextV)
		}
		return {k: ks, size, v: vs}
	}

	attendToken(qh, kh, vh, size) {
		const out = []
		for (let h = 0; h < this.nHead; h += 1) {
			const q = qh[h]
			const k = kh[h]
			const v = vh[h]
			const y = new Float32Array(this.headDim)
			const scores = new Float32Array(size)
			let best = -Infinity
			for (let j = 0; j < size; j += 1) {
				const jOff = j * this.headDim
				let dot = 0
				for (let i = 0; i < this.headDim; i += 1) {
					dot += q[i] * k[jOff + i]
				}
				dot *= this.scale
				scores[j] = dot
				if (dot > best) {
					best = dot
				}
			}
			let total = 0
			for (let j = 0; j < size; j += 1) {
				const w = Math.exp(scores[j] - best)
				scores[j] = w
				total += w
			}
			for (let i = 0; i < this.headDim; i += 1) {
				let sum = 0
				for (let j = 0; j < size; j += 1) {
					const jOff = j * this.headDim
					sum += scores[j] * v[jOff + i]
				}
				y[i] = sum / total
			}
			out.push(y)
		}
		return this.combineHeads(out, 1)
	}

	attnSeq(x, idx, seq) {
		const pre = "blocks." + idx + ".attn."
		const dim = this.nEmbd * 3
		const w = this.ws[pre + "qkv.weight"]
		const qkv = this.linearSeq(x, seq, dim, this.nEmbd, w)
		const q = new Float32Array(seq * this.nEmbd)
		const k = new Float32Array(seq * this.nEmbd)
		const v = new Float32Array(seq * this.nEmbd)
		for (let s = 0; s < seq; s += 1) {
			const a = s * this.nEmbd
			const b = s * this.nEmbd * 3
			q.set(qkv.subarray(b, b + this.nEmbd), a)
			k.set(qkv.subarray(b + this.nEmbd, b + this.nEmbd * 2), a)
			v.set(qkv.subarray(b + this.nEmbd * 2, b + this.nEmbd * 3), a)
		}
		let qh = this.splitHeads(q, seq)
		let kh = this.splitHeads(k, seq)
		const vh = this.splitHeads(v, seq)
		this.applyRotary(qh, seq, 0)
		this.applyRotary(kh, seq, 0)
		qh = this.normHeads(qh, seq)
		kh = this.normHeads(kh, seq)
		const y = this.attendSeq(qh, kh, vh, seq)
		const proj = this.ws[pre + "proj.weight"]
		const out = this.linearSeq(y, seq, this.nEmbd, this.nEmbd, proj)
		return {cache: {k: kh, size: seq, v: vh}, out}
	}

	attnToken(x, idx, cache) {
		const pre = "blocks." + idx + ".attn."
		const dim = this.nEmbd * 3
		const w = this.ws[pre + "qkv.weight"]
		const qkv = this.linearVec(x, dim, this.nEmbd, w)
		const q = qkv.subarray(0, this.nEmbd)
		const k = qkv.subarray(this.nEmbd, this.nEmbd * 2)
		const v = qkv.subarray(this.nEmbd * 2, this.nEmbd * 3)
		let qh = this.splitHeads(q, 1)
		let kh = this.splitHeads(k, 1)
		const vh = this.splitHeads(v, 1)
		this.applyRotary(qh, 1, cache.size)
		this.applyRotary(kh, 1, cache.size)
		qh = this.normHeads(qh, 1)
		kh = this.normHeads(kh, 1)
		const next = this.appendCache(cache, kh, vh)
		const y = this.attendToken(qh, next.k, next.v, next.size)
		const proj = this.ws[pre + "proj.weight"]
		const out = this.linearVec(y, this.nEmbd, this.nEmbd, proj)
		return {cache: next, out}
	}

	mlpSeq(x, idx, seq) {
		const pre = "blocks." + idx + ".mlp."
		const wideDim = this.nEmbd * 4
		const fc = this.ws[pre + "fc.weight"]
		const wide = this.linearSeq(x, seq, wideDim, this.nEmbd, fc)
		for (let i = 0; i < wide.length; i += 1) {
			const v = Math.max(0, wide[i])
			wide[i] = v * v
		}
		const proj = this.ws[pre + "proj.weight"]
		return this.linearSeq(wide, seq, this.nEmbd, wideDim, proj)
	}

	mlpVec(x, idx) {
		return this.mlpSeq(x, idx, 1)
	}

	forwardPrefix(ids) {
		const seq = ids.length
		let x = this.embed(ids)
		const cache = []
		for (let idx = 0; idx < this.cfg.depth; idx += 1) {
			const a = "blocks." + idx + ".attn_norm.weight"
			const m = "blocks." + idx + ".mlp_norm.weight"
			const ax = this.rmsSeq(x, seq, this.nEmbd, this.ws[a])
			const att = this.attnSeq(ax, idx, seq)
			this.addInto(x, att.out)
			const mx = this.rmsSeq(x, seq, this.nEmbd, this.ws[m])
			const ml = this.mlpSeq(mx, idx, seq)
			this.addInto(x, ml)
			cache.push(att.cache)
		}
		const norm = this.rmsSeq(x, seq, this.nEmbd, this.ws["norm.weight"])
		const lm = this.ws["lm_head.weight"]
		const vocab = this.cfg.vocab_size
		const logits = this.linearSeq(norm, seq, vocab, this.nEmbd, lm)
		const off = (seq - 1) * this.cfg.vocab_size
		const end = off + this.cfg.vocab_size
		return {cache, logits: logits.subarray(off, end)}
	}

	forwardToken(id, cache) {
		let x = this.embed([id])
		const next = []
		for (let idx = 0; idx < this.cfg.depth; idx += 1) {
			const a = "blocks." + idx + ".attn_norm.weight"
			const m = "blocks." + idx + ".mlp_norm.weight"
			const ax = this.rmsVec(x, this.nEmbd, this.ws[a])
			const att = this.attnToken(ax, idx, cache[idx])
			this.addInto(x, att.out)
			const mx = this.rmsVec(x, this.nEmbd, this.ws[m])
			const ml = this.mlpVec(mx, idx)
			this.addInto(x, ml)
			next.push(att.cache)
		}
		const norm = this.rmsVec(x, this.nEmbd, this.ws["norm.weight"])
		const lm = this.ws["lm_head.weight"]
		const vocab = this.cfg.vocab_size
		const logits = this.linearVec(norm, vocab, this.nEmbd, lm)
		return {cache: next, logits}
	}

	pickGreedyId(logits, allowed, rng) {
		let best = -Infinity
		let ids = []
		for (const id of allowed) {
			const value = logits[id]
			if (value > best) {
				best = value
				ids = [id]
				continue
			}
			if (value === best) {
				ids.push(id)
			}
		}
		return ids[rng.int(ids.length)]
	}

	decode(text, seed, pickId) {
		const prefix = this.encodeText(text)
		prefix.push(this.tok.sep_id)
		let node = this.trie
		let room = []
		let state = this.forwardPrefix(prefix)
		const rng = new Rng(seed)
		while (true) {
			const id = pickId.call(this, state.logits, node.allowed, rng)
			if (id === this.tok.eos_id) {
				return this.decodeText(room)
			}
			room.push(id)
			node = node.children[id]
			state = this.forwardToken(id, state.cache)
		}
	}

	predictRoom(text, seed) {
		return this.decode(text, seed, this.pickGreedyId)
	}

	levenshtein(left, right, limit) {
		if (limit != null) {
			const gap = Math.abs(left.length - right.length)
			if (gap > limit) {
				return limit + 1
			}
		}
		let prev = []
		for (let i = 0; i <= right.length; i += 1) {
			prev.push(i)
		}
		for (let i = 1; i <= left.length; i += 1) {
			const next = [i]
			for (let j = 1; j <= right.length; j += 1) {
				const ins = next[next.length - 1] + 1
				const del = prev[j] + 1
				const rep = prev[j - 1] + (left[i - 1] !== right[j - 1])
				next.push(Math.min(ins, del, rep))
			}
			prev = next
		}
		const dist = prev[prev.length - 1]
		if (limit != null && dist > limit) {
			return limit + 1
		}
		return dist
	}

	damerau(left, right, limit) {
		if (limit != null) {
			const gap = Math.abs(left.length - right.length)
			if (gap > limit) {
				return limit + 1
			}
		}
		const inf = left.length + right.length
		const last = {}
		const rows = left.length + 2
		const cols = right.length + 2
		const table = Array.from({length: rows}, () => Array(cols).fill(inf))
		table[0][0] = inf
		for (let i = 0; i <= left.length; i += 1) {
			table[i + 1][0] = inf
			table[i + 1][1] = i
		}
		for (let j = 0; j <= right.length; j += 1) {
			table[0][j + 1] = inf
			table[1][j + 1] = j
		}
		for (let i = 1; i <= left.length; i += 1) {
			let match = 0
			for (let j = 1; j <= right.length; j += 1) {
				const seen = last[right[j - 1]] || 0
				const cost = Number(left[i - 1] !== right[j - 1])
				if (!cost) {
					match = j
				}
				const a = table[i][j] + cost
				const b = table[i + 1][j] + 1
				const c = table[i][j + 1] + 1
				let d = table[seen][match]
				d += i - seen + j - match - 1
				table[i + 1][j + 1] = Math.min(a, b, c, d)
			}
			last[left[i - 1]] = i
		}
		const dist = table[rows - 1][cols - 1]
		if (limit != null && dist > limit) {
			return limit + 1
		}
		return dist
	}

	nearestAddress(text, seed, fn) {
		let best = null
		let rooms = []
		const rng = new Rng(seed)
		for (const room of this.rooms) {
			const dist = fn.call(this, text, room, best)
			if (best == null || dist < best) {
				best = dist
				rooms = [room]
				continue
			}
			if (dist === best) {
				rooms.push(room)
			}
		}
		const room = rooms[rng.int(rooms.length)]
		return this.roomMap[room]
	}

	solve(text) {
		const input = this.normalize(text)
		const out = {input}
		out.identity = this.roomMap[input] || ""
		out.levenshtein = this.nearestAddress(input, this.seed, this.levenshtein)
		out.damerau_levenshtein = this.nearestAddress(input, this.seed, this.damerau)
		const exact = this.roomSet.has(input)
		const room = exact ? input : this.predictRoom(input, this.seed)
		out.ours_room = room
		out.ours = this.roomMap[room] || ""
		out.final = out.ours
		return out
	}
}


let model = null


async function boot() {
	try {
		const a = await fetch("./assets.json")
		const b = await fetch("./weights.bin")
		if (!a.ok || !b.ok) {
			throw new Error("Run code/export.py to create assets.json and weights.bin.")
		}
		const assets = await a.json()
		const buf = await b.arrayBuffer()
		model = new Model(assets, buf)
		ui.run.disabled = false
		ui.status.textContent = "Ready."
		ui.input.focus()
	} catch (err) {
		ui.status.textContent = String(err.message || err)
	}
}


function render(result) {
	ui.out.textContent = JSON.stringify(result, null, 2)
}


ui.form.addEventListener("submit", event => {
	event.preventDefault()
	if (!model) {
		return
	}
	const start = performance.now()
	const result = model.solve(ui.input.value)
	result.latency_ms = performance.now() - start
	render(result)
})


boot()
