export type TrainingRole = "system" | "user" | "assistant";
export type TrainingMessage = { role: TrainingRole; content: string };
export type TrainingExample = { messages: TrainingMessage[] };
export type DatasetValidation = { valid: boolean; examples: number; messages: number; characters: number; errors: string[] };
