import type { RuleGroupType, RuleType } from "react-querybuilder";

export type OverrideField = "model" | "request_type";

export const REQUEST_TYPE_VALUES = [
	{ name: "text_completion", label: "Text Completion" },
	{ name: "chat_completion", label: "Chat Completion" },
	{ name: "text_completion_stream", label: "Text Completion (Stream)" },
	{ name: "chat_completion_stream", label: "Chat Completion (Stream)" },
	{ name: "responses", label: "Responses" },
	{ name: "responses_stream", label: "Responses (Stream)" },
	{ name: "embedding", label: "Embeddings" },
	{ name: "image_generation", label: "Image Generation" },
	{ name: "speech", label: "Speech" },
	{ name: "transcription", label: "Transcription" },
	{ name: "count_tokens", label: "Count Tokens" },
] as const;

export const OVERRIDE_FIELD_OPTIONS = [
	{ name: "model" as const, label: "Model" },
	{ name: "request_type" as const, label: "Request Type" },
];

export const MODEL_OPERATORS = [
	{ name: "==", label: "equals" },
	{ name: "!=", label: "not equals" },
	{ name: "contains", label: "contains" },
	{ name: "startsWith", label: "starts with" },
	{ name: "endsWith", label: "ends with" },
	{ name: "matches", label: "matches (regex)" },
];

export const REQUEST_TYPE_OPERATORS = [
	{ name: "==", label: "equals" },
	{ name: "!=", label: "not equals" },
];

export function getOperatorsForField(field: OverrideField) {
	return field === "model" ? MODEL_OPERATORS : REQUEST_TYPE_OPERATORS;
}

function escapeString(s: string): string {
	return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");
}

function ruleToCEL(rule: RuleType): string {
	const field = rule.field as string;
	const operator = rule.operator as string;
	const value = rule.value as string;
	const escaped = escapeString(value);

	if (operator === "==" || operator === "!=") {
		return `${field} ${operator} "${escaped}"`;
	}
	return `${field}.${operator}("${escaped}")`;
}

export function convertRuleGroupToCEL(group: RuleGroupType): string {
	if (!group.rules || group.rules.length === 0) return "";

	const expressions = group.rules
		.map((item) => {
			if ("rules" in item) {
				const nested = convertRuleGroupToCEL(item as RuleGroupType);
				return nested ? `(${nested})` : "";
			}
			const rule = item as RuleType;
			if (!rule.value || (typeof rule.value === "string" && !rule.value.trim())) {
				return "";
			}
			return ruleToCEL(rule);
		})
		.filter((expr) => expr !== "");

	if (expressions.length === 0) return "";
	if (expressions.length === 1) return expressions[0];

	const joiner = group.combinator === "or" ? " || " : " && ";
	return expressions.join(joiner);
}

const SIMPLE_CONDITION_RE = /^(model|request_type)\s*(==|!=)\s*["']([^"']*)["']$/;
const METHOD_CONDITION_RE = /^(model|request_type)\.(contains|startsWith|endsWith|matches)\(["']([^"']*)["']\)$/;

function parseSingleRule(expr: string): RuleType | null {
	const trimmed = expr.trim();
	let m = trimmed.match(SIMPLE_CONDITION_RE);
	if (m) {
		return { field: m[1], operator: m[2], value: m[3] };
	}
	m = trimmed.match(METHOD_CONDITION_RE);
	if (m) {
		return { field: m[1], operator: m[2], value: m[3] };
	}
	return null;
}

function splitByCombinator(expr: string, combinator: "&&" | "||"): string[] {
	const parts: string[] = [];
	let current = "";
	let depth = 0;

	for (let i = 0; i < expr.length; i++) {
		const char = expr[i];
		if (char === "(") depth++;
		else if (char === ")") depth--;

		if (depth === 0 && expr.slice(i, i + 2) === combinator) {
			parts.push(current.trim());
			current = "";
			i++;
		} else {
			current += char;
		}
	}
	if (current.trim()) parts.push(current.trim());
	return parts;
}

export function convertCELToRuleGroup(cel: string): RuleGroupType | null {
	const trimmed = cel.trim();
	if (!trimmed) return { combinator: "and", rules: [] };

	// Remove outer parentheses if present
	let expr = trimmed;
	if (expr.startsWith("(") && expr.endsWith(")")) {
		let depth = 0;
		let isOuterParen = true;
		for (let i = 0; i < expr.length; i++) {
			if (expr[i] === "(") depth++;
			else if (expr[i] === ")") depth--;
			if (depth === 0 && i < expr.length - 1) {
				isOuterParen = false;
				break;
			}
		}
		if (isOuterParen) expr = expr.slice(1, -1).trim();
	}

	// Try single rule
	const singleRule = parseSingleRule(expr);
	if (singleRule) return { combinator: "and", rules: [singleRule] };

	// Try splitting by OR first (lower precedence)
	const orParts = splitByCombinator(expr, "||");
	if (orParts.length > 1) {
		const rules = orParts
			.map((part) => {
				const nested = convertCELToRuleGroup(part);
				if (!nested) return null;
				if (nested.rules.length === 1 && !("rules" in nested.rules[0])) {
					return nested.rules[0];
				}
				return nested;
			})
			.filter((r): r is RuleType | RuleGroupType => r !== null);

		if (rules.length > 0) return { combinator: "or", rules };
	}

	// Try splitting by AND
	const andParts = splitByCombinator(expr, "&&");
	if (andParts.length > 1) {
		const rules = andParts
			.map((part) => {
				const nested = convertCELToRuleGroup(part);
				if (!nested) return null;
				if (nested.rules.length === 1 && !("rules" in nested.rules[0])) {
					return nested.rules[0];
				}
				return nested;
			})
			.filter((r): r is RuleType | RuleGroupType => r !== null);

		if (rules.length > 0) return { combinator: "and", rules };
	}

	return null;
}

export function emptyRuleGroup(): RuleGroupType {
	return { combinator: "and", rules: [] };
}

export function defaultRule(): RuleType {
	return { field: "model", operator: "==", value: "" };
}
