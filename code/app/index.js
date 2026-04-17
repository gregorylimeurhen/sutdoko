"use strict"

const EPS = 1.1920928955078125e-7

const ui = {
	input: document.getElementById("input"),
	output: document.getElementById("output"),
}
class Model {
	constructor(assets, buf) {
		this.aliases = []
		this.cache = {cache: null, ids: [], text: ""}
		this.cfg = assets.config
		this.direct = {}
		this.keys = []
		this.roomLookup = assets.room_lookup
		this.rooms = assets.rooms
		this.roomKeys = {}
		this.roomRows = []
		this.solveCache = new Map()
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
		for (const [source, target] of assets.aliases || []) {
			const left = this.rawText(source)
			const right = this.rawText(target)
			const leftWord = this.hasWord(left)
			const rightWord = this.hasWord(right)
			const gap = right || " "
			if (!left) {
				continue
			}
			if (!leftWord) {
				this.setDirect(left, gap)
				continue
			}
			if (!rightWord) {
				if (right) {
					this.setDirect(right, this.basicKey(left))
				}
				continue
			}
			const from = this.basicWords(right)
			const to = this.basicKey(left)
			if (!from.length || !to) {
				continue
			}
			this.aliases.push([from, to])
		}
		this.aliases.sort((left, right) => {
			const a = left[0].length
			const b = right[0].length
			if (a !== b) {
				return b - a
			}
			return left[1].localeCompare(right[1])
		})
		this.keys = Object.keys(this.direct)
		this.keys.sort((left, right) => right.length - left.length)
		for (const room of this.rooms) {
			const key = this.normalize(room)
			if (!this.roomKeys[key]) {
				this.roomKeys[key] = []
			}
			this.roomKeys[key].push(room)
			this.roomRows.push([room, key, this.charHist(key)])
		}
		for (const key of Object.keys(this.roomKeys)) {
			this.roomKeys[key].sort((left, right) => left.localeCompare(right))
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

	rawText(text) {
		return text.trim().toLowerCase()
	}

	hasWord(text) {
		return /[a-z0-9]/.test(text)
	}

	setDirect(source, target) {
		const prev = this.direct[source]
		if (!prev || target.length > prev.length) {
			this.direct[source] = target
		}
	}

	isLetter(char) {
		return char >= "a" && char <= "z"
	}

	isDigit(char) {
		return char >= "0" && char <= "9"
	}

	splitAlnum(text) {
		let out = ""
		let prev = ""
		for (const char of text) {
			const left = this.isLetter(prev) && this.isDigit(char)
			const right = this.isDigit(prev) && this.isLetter(char)
			if (out && (left || right)) {
				out += " "
			}
			out += char
			prev = char
		}
		return out
	}

	basicKey(text) {
		let out = this.rawText(text)
		for (const key of this.keys) {
			out = out.split(key).join(this.direct[key])
		}
		out = this.splitAlnum(out)
		out = out.replace(/[^a-z0-9]+/g, " ")
		return out.trim().replace(/\s+/g, " ")
	}

	basicWords(text) {
		const key = this.basicKey(text)
		if (!key) {
			return []
		}
		return key.split(" ")
	}

	expandAliases(words) {
		let out = []
		let start = 0
		while (start < words.length) {
			let hit = null
			for (const row of this.aliases) {
				const end = start + row[0].length
				if (end > words.length) {
					continue
				}
				let okay = true
				for (let index = 0; index < row[0].length; index += 1) {
					if (words[start + index] !== row[0][index]) {
						okay = false
						break
					}
				}
				if (!okay) {
					continue
				}
				hit = row
				break
			}
			if (!hit) {
				out.push(words[start])
				start += 1
				continue
			}
			out.push(hit[1])
			start += hit[0].length
		}
		return out.join(" ")
	}

	normalize(text) {
		return this.expandAliases(this.basicWords(text))
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

	charHist(text) {
		const hist = {}
		for (const char of text) {
			hist[char] = (hist[char] || 0) + 1
		}
		return hist
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

	allowedLogps(logits, allowed) {
		let best = -Infinity
		for (const id of allowed) {
			const value = logits[id]
			if (value > best) {
				best = value
			}
		}
		let total = 0
		let rows = []
		for (const id of allowed) {
			const value = Math.exp(logits[id] - best)
			rows.push([id, value])
			total += value
		}
		for (const row of rows) {
			row[1] = Math.log(row[1] / total)
		}
		return rows
	}

	topRoomRows(rows, rev) {
		rows.sort((left, right) => {
			if (left[0] !== right[0]) {
				return rev ? right[0] - left[0] : left[0] - right[0]
			}
			return left[1].localeCompare(right[1])
		})
		return rows.slice(0, 2).map(row => row[1])
	}

	inputState(text) {
		const ids = this.encodeText(text)
		if (!ids.length) {
			return {cache: null, ids, text}
		}
		if (text === this.cache.text) {
			return this.cache
		}
		const prev = this.cache
		if (prev.text && text.startsWith(prev.text) && prev.cache) {
			let cache = prev.cache
			let next = prev.ids.slice()
			for (const id of ids.slice(next.length)) {
				cache = this.forwardToken(id, cache).cache
				next.push(id)
			}
			this.cache = {cache, ids: next, text}
			return this.cache
		}
		const state = this.forwardPrefix(ids)
		this.cache = {cache: state.cache, ids, text}
		return this.cache
	}

	decodeBeam(text, width) {
		const prefix = this.inputState(text)
		const state = this.forwardToken(this.tok.sep_id, prefix.cache)
		let beams = [{
			cache: state.cache,
			logits: state.logits,
			node: this.trie,
			room: [],
			score: 0,
		}]
		let done = []
		while (beams.length) {
			let next = []
			for (const beam of beams) {
				const rows = this.allowedLogps(beam.logits, beam.node.allowed)
				for (const [id, logp] of rows) {
					const score = beam.score + logp
					if (id === this.tok.eos_id) {
						const text = this.decodeText(beam.room)
						done.push([score, text])
						continue
					}
					const room = beam.room.concat(id)
					next.push({id, room, score, src: beam})
				}
			}
			next.sort((left, right) => {
				if (left.score !== right.score) {
					return right.score - left.score
				}
				const a = this.decodeText(left.room)
				const b = this.decodeText(right.room)
				return a.localeCompare(b)
			})
			next = next.slice(0, width)
			beams = next.map(row => {
				const src = row.src
				const state = this.forwardToken(row.id, src.cache)
				return {
					cache: state.cache,
					logits: state.logits,
					node: src.node.children[row.id],
					room: row.room,
					score: row.score,
				}
			})
		}
		done.sort((left, right) => {
			if (left[0] !== right[0]) {
				return right[0] - left[0]
			}
			return left[1].localeCompare(right[1])
		})
		return done.slice(0, 2).map(row => row[1])
	}

	nearestRooms(text, fn) {
		let rows = []
		for (const [room, key] of this.roomRows) {
			rows.push([fn.call(this, text, key), room])
		}
		return this.topRoomRows(rows)
	}

	bestRooms(text, fn) {
		let rows = []
		for (const [room, key] of this.roomRows) {
			rows.push([fn.call(this, text, key), room])
		}
		return this.topRoomRows(rows, true)
	}

	histRooms(text) {
		const left = this.charHist(text)
		let rows = []
		for (const [room, _, right] of this.roomRows) {
			rows.push([this.histScore(left, right), room])
		}
		return this.topRoomRows(rows, true)
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

	lcs(left, right, limit) {
		if (right.length > left.length) {
			[left, right] = [right, left]
		}
		if (limit != null && right.length < limit) {
			return -1
		}
		let prev = Array(right.length + 1).fill(0)
		for (let i = 1; i <= left.length; i += 1) {
			const next = [0]
			for (let j = 1; j <= right.length; j += 1) {
				let value = prev[j]
				if (next[next.length - 1] > value) {
					value = next[next.length - 1]
				}
				if (left[i - 1] === right[j - 1]) {
					const match = prev[j - 1] + 1
					if (match > value) {
						value = match
					}
				}
				next.push(value)
			}
			prev = next
		}
		return prev[prev.length - 1]
	}

	histScore(left, right) {
		let score = 0
		for (const char of Object.keys(left)) {
			score += Math.min(left[char], right[char] || 0)
		}
		return score
	}

	solve(text) {
		const input = this.normalize(text)
		if (!input) {
			return []
		}
		if (this.solveCache.has(input)) {
			return this.solveCache.get(input)
		}
		let out = []
		const exact = this.roomKeys[input]
		if (exact) {
			out.push(...exact)
		}
		out.push(...this.nearestRooms(input, this.levenshtein))
		out.push(...this.nearestRooms(input, this.damerau))
		out.push(...this.bestRooms(input, this.lcs))
		out.push(...this.histRooms(input))
		try {
			out.push(...this.decodeBeam(input, 2))
		} catch (_) {}
		out = Array.from(new Set(out)).sort((a, b) => a.localeCompare(b))
		out = out.slice(0, 10).map(room => [room, this.roomLookup[room] || ""])
		this.solveCache.set(input, out)
		return out
	}
}


let model = null


async function boot() {
	try {
		const a = await fetch("./assets.json")
		const b = await fetch("./weights.bin")
		if (!a.ok || !b.ok) {
			return
		}
		const assets = await a.json()
		const buf = await b.arrayBuffer()
		model = new Model(assets, buf)
		ui.input.focus()
		update()
	} catch (_) {
	}
}


function render(rows) {
	ui.output.replaceChildren()
	for (let i = 0; i < 10; i += 1) {
		const tr = document.createElement("tr")
		const name = document.createElement("td")
		const address = document.createElement("td")
		const row = rows[i] || ["", ""]
		name.textContent = row[0]
		address.textContent = row[1]
		tr.append(name, address)
		ui.output.append(tr)
	}
}


function update() {
	if (!model) {
		render([])
		return
	}
	try {
		render(model.solve(ui.input.value))
	} catch (_) {
		render([])
	}
}


ui.input.addEventListener("input", update)


boot()
