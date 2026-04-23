"use strict"

const EPS = 1.1920928955078125e-7

let model = null
let ready = null


class Model {
	constructor(assets, buf) {
		this.aliases = []
		this.cache = {cache: null, ids: [], text: ""}
		this.cfg = assets.config
		this.roomLookup = assets.room_lookup
		this.rooms = assets.rooms
		this.roomCount = this.rooms.length
		this.roomOrder = []
		this.roomRows = []
		this.roomSet = new Set()
		this.solveCache = new Map()
		this.tok = assets.tokenizer
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
			const from = this.aliasWords(target)
			const to = this.preAliasText(source)
			if (!from.length || !to) {
				continue
			}
			this.aliases.push([from, to])
		}
		this.aliases.sort((left, right) => {
			if (left[0].length !== right[0].length) {
				return right[0].length - left[0].length
			}
			return left[1].localeCompare(right[1])
		})
			for (let index = 0; index < this.rooms.length; index += 1) {
				const room = this.rooms[index]
				const key = this.normalize(room)
				const ids = this.encodeText(room)
				const subs = this.substringSet(key)
				this.roomOrder.push(index)
				this.roomRows.push({ids, index, key, subs})
				this.roomSet.add(key)
			}
		for (const name of Object.keys(assets.tensors)) {
			const info = assets.tensors[name]
			this.ws[name] = new Float32Array(buf, info.offset, info.size)
		}
		this.rot = this.buildRotary()
		this.roomTrie = this.buildRoomTrie(this.roomRows)
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

	isLetter(char) {
		return char >= "a" && char <= "z"
	}

	isDigit(char) {
		return char >= "0" && char <= "9"
	}

	depunctuate(text) {
		return text.replace(/[^A-Za-z0-9\s]+/g, "")
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

	preAliasText(text) {
		text = text.trim()
		text = this.depunctuate(text)
		text = text.toLowerCase()
		text = this.splitAlnum(text)
		return text.trim().replace(/\s+/g, " ")
	}

	aliasWords(text) {
		text = this.preAliasText(text)
		if (!text) {
			return []
		}
		return text.split(" ")
	}

	expandAliases(words) {
		const out = []
		let start = 0
		while (start < words.length) {
			let hit = null
			for (const row of this.aliases) {
				const end = start + row[0].length
				if (end > words.length) {
					continue
				}
				let okay = true
				for (let i = 0; i < row[0].length; i += 1) {
					if (words[start + i] !== row[0][i]) {
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
		return this.expandAliases(this.aliasWords(text))
	}

	encodeText(text) {
		const ids = []
		const unk = this.tok.unk_id
		for (const char of text) {
			ids.push(this.stoi[char] ?? unk)
		}
		return ids
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
			if (logits[id] > best) {
				best = logits[id]
			}
		}
		let total = 0
		const rows = []
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

	commonPrefix(left, right) {
		const len = Math.min(left.length, right.length)
		let i = 0
		while (i < len && left[i] === right[i]) {
			i += 1
		}
		return i
	}

	substringSet(text) {
		const out = new Set()
		for (let start = 0; start < text.length; start += 1) {
			for (let stop = start + 1; stop <= text.length; stop += 1) {
				out.add(text.slice(start, stop))
			}
		}
		return out
	}

		buildRoomTrie(rows) {
		const root = {allowed: [], children: {}, roomIndex: -1}
		for (const row of rows) {
			let node = root
			for (const id of row.ids) {
				let next = node.children[id]
				if (!next) {
					next = {allowed: [], children: {}, roomIndex: -1}
					node.children[id] = next
				}
				node = next
			}
			node.children[this.tok.eos_id] = {
				allowed: [],
				children: {},
				roomIndex: row.index,
			}
		}
		const stack = [[root, false]]
		while (stack.length) {
			const [node, seen] = stack.pop()
			if (!seen) {
				stack.push([node, true])
				for (const id of Object.keys(node.children)) {
					stack.push([node.children[id], false])
				}
				continue
			}
			const ids = Object.keys(node.children).map(Number)
			ids.sort((left, right) => this.vocab[left].localeCompare(this.vocab[right]))
			node.allowed = ids
		}
		return root
	}

	assignRanks(order, same) {
		const ranks = new Uint16Array(this.roomCount)
		let rank = 1
		for (let i = 0; i < order.length; i += 1) {
			if (i && !same(order[i - 1], order[i])) {
				rank = i + 1
			}
			ranks[order[i]] = rank
		}
		return ranks
	}

	rankPrefix(text) {
		const scores = new Uint16Array(this.roomCount)
		for (const row of this.roomRows) {
			scores[row.index] = this.commonPrefix(text, row.key)
		}
		const order = this.roomOrder.slice()
		order.sort((left, right) => scores[right] - scores[left] || left - right)
		return this.assignRanks(order, (left, right) => scores[left] === scores[right])
	}

	substringCounts(left, right) {
		let hit = 0
		let small = left
		let big = right
		if (right.size < left.size) {
			small = right
			big = left
		}
		for (const value of small) {
			if (big.has(value)) {
				hit += 1
			}
		}
		const union = left.size + right.size - hit
		return [hit, union || 1]
	}

	rankJaccard(leftSubs) {
		const nums = new Uint16Array(this.roomCount)
		const dens = new Uint16Array(this.roomCount)
		for (const row of this.roomRows) {
			const [num, den] = this.substringCounts(leftSubs, row.subs)
			nums[row.index] = num
			dens[row.index] = den
		}
		const order = this.roomOrder.slice()
		order.sort((left, right) => {
			const a = nums[left] * dens[right]
			const b = nums[right] * dens[left]
			return b - a || left - right
		})
		return this.assignRanks(order, (left, right) => {
			return nums[left] * dens[right] === nums[right] * dens[left]
		})
	}

	rankDamerau(text) {
		const scores = new Uint16Array(this.roomCount)
		for (const row of this.roomRows) {
			scores[row.index] = this.damerau(text, row.key)
		}
		const order = this.roomOrder.slice()
		order.sort((left, right) => scores[left] - scores[right] || left - right)
		return this.assignRanks(order, (left, right) => scores[left] === scores[right])
	}

	buildIndexTrie(indices) {
		const root = {children: {}}
		for (const index of indices) {
			let node = root
			for (const id of this.roomRows[index].ids) {
				let next = node.children[id]
				if (!next) {
					next = {children: {}, roomIndex: -1}
					node.children[id] = next
				}
				node = next
			}
			node.children[this.tok.eos_id] = {children: {}, roomIndex: index}
		}
		return root
	}

	scoreGroupInto(fullNode, groupNode, cache, logits, score, scores) {
		for (const [id, logp] of this.allowedLogps(logits, fullNode.allowed)) {
			const child = groupNode.children[id]
			if (!child) {
				continue
			}
			const nextScore = score + logp
			if (id === this.tok.eos_id) {
				scores[child.roomIndex] = nextScore
				continue
			}
			const next = this.forwardToken(id, cache)
			this.scoreGroupInto(fullNode.children[id], child, next.cache, next.logits, nextScore, scores)
		}
	}

	oursStart(text) {
		const prefix = this.inputState(text)
		return this.forwardToken(this.tok.sep_id, prefix.cache)
	}

		orderOursGroup(start, indices) {
			const scores = new Float64Array(this.roomCount)
			scores.fill(-Infinity)
		const groupTrie = this.buildIndexTrie(indices)
		this.scoreGroupInto(this.roomTrie, groupTrie, start.cache, start.logits, 0, scores)
		const order = indices.slice()
		order.sort((left, right) => {
				if (scores[left] !== scores[right]) {
					return scores[right] - scores[left]
				}
				return left - right
			})
			return order
		}

	rankRooms(text) {
		const leftSubs = this.substringSet(text)
		const prefix = this.rankPrefix(text)
		const jaccard = this.rankJaccard(leftSubs)
		const damerau = this.rankDamerau(text)
		const order = this.roomOrder.slice()
		order.sort((left, right) => {
			if (prefix[left] !== prefix[right]) {
				return prefix[left] - prefix[right]
			}
			if (jaccard[left] !== jaccard[right]) {
				return jaccard[left] - jaccard[right]
			}
			if (damerau[left] !== damerau[right]) {
				return damerau[left] - damerau[right]
			}
			return left - right
		})
		const picks = []
		let start = null
		let i = 0
		while (i < order.length && picks.length < 3) {
			let j = i + 1
			while (j < order.length) {
				const left = order[i]
				const right = order[j]
				if (prefix[left] !== prefix[right]) {
					break
				}
				if (jaccard[left] !== jaccard[right]) {
					break
				}
					if (damerau[left] !== damerau[right]) {
						break
					}
					j += 1
				}
				const group = order.slice(i, j)
				let rows = group
				if (group.length !== 1) {
					start = start || this.oursStart(text)
					rows = this.orderOursGroup(start, group)
				}
				for (const index of rows) {
					picks.push(index)
					if (picks.length === 3) {
						break
					}
				}
				i = j
			}
		const out = []
		for (const index of picks) {
			const room = this.rooms[index]
			out.push([room, this.roomLookup[room] || ""])
		}
		return out
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
			const next = prev.ids.slice()
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

	damerau(left, right, limit) {
		if (limit != null && Math.abs(left.length - right.length) > limit) {
			return limit + 1
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

	solve(text) {
		const input = this.normalize(text)
		if (!input) {
			return []
		}
		const hit = this.solveCache.get(input)
		if (hit) {
			return hit
		}
		if (this.roomSet.has(input)) {
			const out = [[input, this.roomLookup[input] || ""]]
			this.solveCache.set(input, out)
			return out
		}
		const out = this.rankRooms(input)
		this.solveCache.set(input, out)
		return out
	}
}


async function boot() {
	const assetsResponse = await fetch("./assets.json")
	const weightsResponse = await fetch("./weights.bin")
	if (!assetsResponse.ok || !weightsResponse.ok) {
		throw new Error("missing assets")
	}
	const assets = await assetsResponse.json()
	const buf = await weightsResponse.arrayBuffer()
	model = new Model(assets, buf)
}


ready = boot().then(() => {
	self.postMessage({type: "ready"})
})


self.addEventListener("message", async ({data}) => {
	if (!data || data.type !== "solve") {
		return
	}
	try {
		await ready
		const rows = model ? model.solve(data.text || "") : []
		self.postMessage({rows, text: data.text || "", type: "result"})
	} catch (_) {
		self.postMessage({text: data.text || "", type: "error"})
	}
})
