"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { TagInput } from "@/components/ui/tagInput";
import { getErrorMessage, setProviderFormDirtyState, useAppDispatch } from "@/lib/store";
import { useUpdateProviderMutation } from "@/lib/store/apis/providersApi";
import { ModelProvider, RequestOverride } from "@/lib/types/config";
import {
	convertCELToRuleGroup,
	convertRuleGroupToCEL,
	emptyRuleGroup,
	MODEL_OPERATORS,
	OVERRIDE_FIELD_OPTIONS,
	REQUEST_TYPE_OPERATORS,
	REQUEST_TYPE_VALUES,
} from "@/lib/utils/celConverterOverrides";
import { ActionButton } from "@/app/workspace/routing-rules/components/celBuilder/actionButton";
import { CombinatorSelector } from "@/app/workspace/routing-rules/components/celBuilder/combinatorSelector";
import { FieldSelector } from "@/app/workspace/routing-rules/components/celBuilder/fieldSelector";
import { OperatorSelector } from "@/app/workspace/routing-rules/components/celBuilder/operatorSelector";
import { QueryBuilderWrapper } from "@/app/workspace/routing-rules/components/celBuilder/queryBuilderWrapper";
import { ValueEditor } from "@/app/workspace/routing-rules/components/celBuilder/valueEditor";
import { RbacOperation, RbacResource, useRbac } from "@enterprise/lib";
import { Code, Eye, Plus, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Field, QueryBuilder, RuleGroupType } from "react-querybuilder";
import "react-querybuilder/dist/query-builder.css";
import { toast } from "sonner";

interface RequestOverridesFormFragmentProps {
	provider: ModelProvider;
}

type KVEntry = { key: string; value: string };

function toKVEntries(obj: Record<string, unknown> | undefined): KVEntry[] {
	if (!obj) return [];
	return Object.entries(obj).map(([key, value]) => ({
		key,
		value: typeof value === "string" ? value : JSON.stringify(value),
	}));
}

function fromKVEntries(entries: KVEntry[]): Record<string, unknown> {
	const result: Record<string, unknown> = {};
	for (const { key, value } of entries) {
		if (!key.trim()) continue;
		try {
			result[key.trim()] = JSON.parse(value);
		} catch {
			result[key.trim()] = value;
		}
	}
	return result;
}

type MatchMode = "visual" | "cel";

interface RuleState {
	match: string;
	matchMode: MatchMode;
	ruleGroup: RuleGroupType;
	setEntries: KVEntry[];
	defaultsEntries: KVEntry[];
	remove: string[];
}

function ruleToState(rule: RequestOverride): RuleState {
	const parsed = convertCELToRuleGroup(rule.match);
	return {
		match: rule.match,
		matchMode: parsed !== null ? "visual" : "cel",
		ruleGroup: parsed ?? emptyRuleGroup(),
		setEntries: toKVEntries(rule.set),
		defaultsEntries: toKVEntries(rule.defaults),
		remove: rule.remove ?? [],
	};
}

function stateToRule(state: RuleState): RequestOverride {
	const match = state.matchMode === "visual" ? convertRuleGroupToCEL(state.ruleGroup) : state.match;
	const rule: RequestOverride = { match };
	const set = fromKVEntries(state.setEntries);
	if (Object.keys(set).length > 0) rule.set = set;
	const defaults = fromKVEntries(state.defaultsEntries);
	if (Object.keys(defaults).length > 0) rule.defaults = defaults;
	if (state.remove.length > 0) rule.remove = state.remove;
	return rule;
}

// --- KV Editor ---

interface KVEditorProps {
	entries: KVEntry[];
	onChange: (entries: KVEntry[]) => void;
	disabled: boolean;
	testIdPrefix: string;
	valuePlaceholder?: string;
}

function KVEditor({ entries, onChange, disabled, testIdPrefix, valuePlaceholder = "value or JSON" }: KVEditorProps) {
	const addEntry = () => onChange([...entries, { key: "", value: "" }]);
	const removeEntry = (i: number) => onChange(entries.filter((_, idx) => idx !== i));
	const updateEntry = (i: number, field: "key" | "value", val: string) => {
		const next = [...entries];
		next[i] = { ...next[i], [field]: val };
		onChange(next);
	};

	return (
		<div className="space-y-1.5">
			{entries.map((entry, i) => (
				<div key={i} className="flex items-center gap-1.5">
					<Input
						className="h-8 flex-1 font-mono text-xs"
						placeholder="key"
						value={entry.key}
						onChange={(e) => updateEntry(i, "key", e.target.value)}
						disabled={disabled}
						data-testid={`${testIdPrefix}-key-${i}`}
					/>
					<Input
						className="h-8 flex-1 font-mono text-xs"
						placeholder={valuePlaceholder}
						value={entry.value}
						onChange={(e) => updateEntry(i, "value", e.target.value)}
						disabled={disabled}
						data-testid={`${testIdPrefix}-value-${i}`}
					/>
					<Button
						type="button"
						variant="ghost"
						size="icon"
						className="h-8 w-8 shrink-0"
						onClick={() => removeEntry(i)}
						disabled={disabled}
						aria-label="Remove entry"
					>
						<X className="h-3.5 w-3.5" />
					</Button>
				</div>
			))}
			<Button
				type="button"
				variant="outline"
				size="sm"
				className="h-8 text-xs"
				onClick={addEntry}
				disabled={disabled}
				data-testid={`${testIdPrefix}-add`}
			>
				<Plus className="mr-1 h-3 w-3" />
				Add
			</Button>
		</div>
	);
}

// --- Visual Match Builder ---

interface MatchBuilderProps {
	rule: RuleState;
	onChange: (patch: Partial<RuleState>) => void;
	disabled: boolean;
	ruleIndex: number;
}

function MatchBuilder({ rule, onChange, disabled, ruleIndex }: MatchBuilderProps) {
	const isVisual = rule.matchMode === "visual";

	const switchToVisual = () => {
		const parsed = convertCELToRuleGroup(rule.match);
		if (parsed) {
			onChange({ matchMode: "visual", ruleGroup: parsed });
		} else {
			toast.warning("Cannot parse this CEL expression into visual mode. Please simplify it or continue editing as CEL.");
		}
	};

	const switchToCEL = () => {
		const cel = convertRuleGroupToCEL(rule.ruleGroup);
		onChange({ matchMode: "cel", match: cel });
	};

	const fields: Field[] = useMemo(
		() =>
			OVERRIDE_FIELD_OPTIONS.map((field) => {
				const baseField = {
					name: field.name,
					label: field.label,
					value: field.name,
				};

				// For request_type, provide values for dropdown
				if (field.name === "request_type") {
					return {
						...baseField,
						valueEditorType: "select" as const,
						values: [...REQUEST_TYPE_VALUES],
					};
				}

				return baseField;
			}),
		[],
	);

	const operators = useMemo(() => {
		const allOps = [...MODEL_OPERATORS, ...REQUEST_TYPE_OPERATORS];
		const unique = Array.from(new Map(allOps.map((op) => [op.name, op])).values());
		return unique.map((op) => ({ name: op.name, label: op.label }));
	}, []);

	const prefix = `provider-request-override-match-${ruleIndex}`;

	return (
		<div className="space-y-2">
			<div className="flex items-center justify-between">
				<Label className="text-xs">Match</Label>
				<Button
					type="button"
					variant="ghost"
					size="sm"
					className="h-6 gap-1 px-2 text-xs"
					onClick={isVisual ? switchToCEL : switchToVisual}
					disabled={disabled}
				>
					{isVisual ? (
						<>
							<Code className="h-3 w-3" />
							CEL
						</>
					) : (
						<>
							<Eye className="h-3 w-3" />
							Visual
						</>
					)}
				</Button>
			</div>

			{isVisual ? (
				<div className="rounded-md border">
					<div className="custom-scrollbar flex w-full flex-col overflow-scroll">
						<QueryBuilderWrapper>
							<QueryBuilder
								fields={fields}
								query={rule.ruleGroup}
								onQueryChange={(q) => onChange({ ruleGroup: q })}
								operators={operators}
								disabled={disabled}
								controlClassnames={{ queryBuilder: "queryBuilder-branches" }}
								controlElements={{
									fieldSelector: FieldSelector,
									operatorSelector: OperatorSelector,
									valueEditor: ValueEditor,
									addRuleAction: ActionButton,
									addGroupAction: ActionButton,
									removeRuleAction: ActionButton,
									removeGroupAction: ActionButton,
									combinatorSelector: CombinatorSelector,
								}}
								translations={{
									addRule: { label: "Add Rule" },
									addGroup: { label: "Add Rule Group" },
								}}
							/>
						</QueryBuilderWrapper>
					</div>
				</div>
			) : (
				<Input
					className="h-8 font-mono text-xs"
					placeholder={`model.contains('claude') && request_type == 'chat_completion'`}
					value={rule.match}
					onChange={(e) => onChange({ match: e.target.value })}
					disabled={disabled}
					data-testid={`${prefix}-cel`}
				/>
			)}
		</div>
	);
}

// --- Main Component ---

export function RequestOverridesFormFragment({ provider }: RequestOverridesFormFragmentProps) {
	const dispatch = useAppDispatch();
	const hasUpdateProviderAccess = useRbac(RbacResource.ModelProvider, RbacOperation.Update);
	const [updateProvider, { isLoading: isUpdatingProvider }] = useUpdateProviderMutation();

	const initialRules = useMemo(() => provider.request_overrides ?? [], [provider.request_overrides]);
	const [rules, setRules] = useState<RuleState[]>(() => initialRules.map(ruleToState));

	useEffect(() => {
		setRules(initialRules.map(ruleToState));
	}, [initialRules, provider.name]);

	const isDirty = useMemo(() => {
		const current = JSON.stringify(rules.map(stateToRule));
		const saved = JSON.stringify(initialRules);
		return current !== saved;
	}, [rules, initialRules]);

	useEffect(() => {
		dispatch(setProviderFormDirtyState(isDirty));
	}, [dispatch, isDirty]);

	const addRule = () =>
		setRules((prev) => [
			...prev,
			{ match: "", matchMode: "visual", ruleGroup: emptyRuleGroup(), setEntries: [], defaultsEntries: [], remove: [] },
		]);

	const removeRule = useCallback((i: number) => {
		setRules((prev) => prev.filter((_, idx) => idx !== i));
	}, []);

	const updateRule = useCallback((i: number, patch: Partial<RuleState>) => {
		setRules((prev) => prev.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
	}, []);

	const onReset = () => setRules(initialRules.map(ruleToState));

	const onSave = async () => {
		try {
			await updateProvider({
				...provider,
				request_overrides: rules.map(stateToRule),
			}).unwrap();
			toast.success("Request overrides updated successfully");
		} catch (err) {
			toast.error("Failed to update request overrides", { description: getErrorMessage(err) });
		}
	};

	return (
		<div className="space-y-4 px-6 pb-6">
			<div className="space-y-1">
				<p className="text-sm font-medium">Request Parameter Overrides</p>
				<p className="text-muted-foreground text-xs">
					Rules are evaluated in order; all matching rules are applied. Each rule matches requests by model and request type. Leave match
					empty to always apply.
				</p>
			</div>

			<div className="space-y-3">
				{rules.length === 0 && (
					<p className="text-muted-foreground rounded-md border border-dashed px-4 py-6 text-center text-xs">
						No override rules. Click &quot;Add Rule&quot; to create one.
					</p>
				)}

				{rules.map((rule, i) => (
					<div key={i} className="space-y-3 rounded-md border p-4">
						<div className="flex items-center justify-between">
							<span className="text-xs font-medium">Rule {i + 1}</span>
							<Button
								type="button"
								variant="ghost"
								size="icon"
								className="h-7 w-7"
								onClick={() => removeRule(i)}
								disabled={!hasUpdateProviderAccess}
								aria-label={`Remove rule ${i + 1}`}
								data-testid={`provider-request-override-remove-rule-${i}`}
							>
								<Trash2 className="h-3.5 w-3.5" />
							</Button>
						</div>

						<MatchBuilder rule={rule} onChange={(patch) => updateRule(i, patch)} disabled={!hasUpdateProviderAccess} ruleIndex={i} />

						<div className="space-y-1">
							<Label className="text-xs">
								Set <span className="text-muted-foreground font-normal">— force-set params (overwrites existing)</span>
							</Label>
							<KVEditor
								entries={rule.setEntries}
								onChange={(setEntries) => updateRule(i, { setEntries })}
								disabled={!hasUpdateProviderAccess}
								testIdPrefix={`provider-request-override-set-${i}`}
							/>
						</div>

						<div className="space-y-1">
							<Label className="text-xs">
								Defaults <span className="text-muted-foreground font-normal">— set only if not already present</span>
							</Label>
							<KVEditor
								entries={rule.defaultsEntries}
								onChange={(defaultsEntries) => updateRule(i, { defaultsEntries })}
								disabled={!hasUpdateProviderAccess}
								testIdPrefix={`provider-request-override-defaults-${i}`}
							/>
						</div>

						<div className="space-y-1">
							<Label className="text-xs">
								Remove <span className="text-muted-foreground font-normal">— param names to delete (Enter or comma to add)</span>
							</Label>
							<TagInput
								value={rule.remove}
								onValueChange={(remove) => updateRule(i, { remove })}
								disabled={!hasUpdateProviderAccess}
								placeholder="e.g. top_p"
								data-testid={`provider-request-override-remove-${i}`}
							/>
						</div>
					</div>
				))}
			</div>

			<Button
				type="button"
				variant="outline"
				size="sm"
				onClick={addRule}
				disabled={!hasUpdateProviderAccess}
				data-testid="provider-request-override-add-rule"
			>
				<Plus className="mr-1.5 h-3.5 w-3.5" />
				Add Rule
			</Button>

			<div className="flex justify-end gap-2">
				<Button
					type="button"
					variant="outline"
					onClick={onReset}
					disabled={!hasUpdateProviderAccess || !isDirty}
					data-testid="provider-request-overrides-reset-button"
				>
					Reset
				</Button>
				<Button
					type="button"
					onClick={onSave}
					isLoading={isUpdatingProvider}
					disabled={!hasUpdateProviderAccess || !isDirty}
					data-testid="provider-request-overrides-save-button"
				>
					Save Request Overrides
				</Button>
			</div>
		</div>
	);
}
