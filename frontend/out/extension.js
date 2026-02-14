"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = require("vscode");
const node_fetch_1 = require("node-fetch");
// Configuration
const API_URL = 'http://localhost:5000';
const INLINE_COMPLETE_DELAY = 300; // ms delay before triggering
let inlineCompletionTimeout;
let lastCompletionPosition;
function activate(context) {
    console.log('Vincent Copilot extension is now active!');
    // Register manual completion command (backward compatibility)
    let manualComplete = vscode.commands.registerCommand('vincentCopilot.complete', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showInformationMessage('No active editor');
            return;
        }
        const document = editor.document;
        const selection = editor.selection;
        const prompt = document.getText(new vscode.Range(new vscode.Position(0, 0), selection.end));
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'Vincent Copilot: Generating completion...'
        }, async () => {
            try {
                const response = await (0, node_fetch_1.default)(`${API_URL}/complete`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, max_new_tokens: 128 })
                });
                if (!response.ok) {
                    const text = await response.text();
                    vscode.window.showErrorMessage(`Vincent Copilot API error (${response.status}): ${text.substring(0, 200)}`);
                    return;
                }
                const data = await response.json();
                if (data.completion) {
                    const newText = data.completion.substring(prompt.length);
                    await editor.edit(editBuilder => {
                        editBuilder.insert(selection.end, newText);
                    });
                }
                else {
                    vscode.window.showErrorMessage('No completion received from Vincent Copilot API.');
                }
            }
            catch (err) {
                vscode.window.showErrorMessage('Error connecting to Vincent Copilot API: ' + err);
            }
        });
    });
    // Register inline completion provider for real-time suggestions
    const inlineProvider = vscode.languages.registerInlineCompletionItemProvider([
        { language: 'python' },
        { language: 'javascript' },
        { language: 'typescript' },
        { language: 'java' },
        { language: 'cpp' },
        { language: 'go' },
        { language: 'rust' },
        { language: 'csharp' },
        { language: 'php' },
        { language: 'ruby' },
        { language: 'kotlin' },
        { language: 'swift' },
        { language: 'scala' },
        { language: 'r' },
        { language: 'sql' },
        { language: 'bash' },
        { language: 'dart' },
        { language: 'elixir' },
        { language: 'perl' },
        { language: 'lua' },
        { language: 'objective-c' },
        { language: 'yaml' },
        { language: 'html' },
        { language: 'css' },
        { language: 'scss' },
        { language: 'haskell' }
    ], {
        async provideInlineCompletionItems(document, position, context, token) {
            // Don't trigger if user is typing too fast or if cancelled
            if (token.isCancellationRequested) {
                return undefined;
            }
            // Get text before and after cursor
            const textBeforeCursor = document.getText(new vscode.Range(new vscode.Position(0, 0), position));
            const textAfterCursor = document.getText(new vscode.Range(position, new vscode.Position(position.line + 10, 0)));
            // Don't trigger on empty documents or very short text
            if (textBeforeCursor.trim().length < 3) {
                return undefined;
            }
            // Detect language
            const language = document.languageId;
            try {
                // Call the /inline-complete endpoint
                const response = await (0, node_fetch_1.default)(`${API_URL}/inline-complete`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prefix: textBeforeCursor,
                        suffix: textAfterCursor.substring(0, 200),
                        language: language,
                        max_new_tokens: 50
                    })
                });
                if (!response.ok) {
                    console.error('Vincent Copilot inline completion error:', response.status);
                    return undefined;
                }
                const data = await response.json();
                if (data.completion && data.completion.trim().length > 0 && data.is_valid) {
                    // Create inline completion item
                    const completionItem = new vscode.InlineCompletionItem(data.completion, new vscode.Range(position, position));
                    lastCompletionPosition = position;
                    return [completionItem];
                }
                return undefined;
            }
            catch (error) {
                console.error('Vincent Copilot connection error:', error);
                return undefined;
            }
        }
    });
    // Register text document change listener for better context
    const changeListener = vscode.workspace.onDidChangeTextDocument((event) => {
        if (!event.document || !vscode.window.activeTextEditor) {
            return;
        }
        // Clear existing timeout
        if (inlineCompletionTimeout) {
            clearTimeout(inlineCompletionTimeout);
        }
        // Set new timeout for inline completion trigger
        inlineCompletionTimeout = setTimeout(() => {
            // Trigger inline completion suggestion
            vscode.commands.executeCommand('editor.action.inlineSuggest.trigger');
        }, INLINE_COMPLETE_DELAY);
    });
    // Register status bar item
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.text = '$(zap) Vincent';
    statusBarItem.tooltip = 'Vincent Copilot is active - Press Tab to accept suggestions';
    statusBarItem.command = 'vincentCopilot.complete';
    statusBarItem.show();
    // Health check on activation
    checkApiHealth(statusBarItem);
    context.subscriptions.push(manualComplete, inlineProvider, changeListener, statusBarItem);
}
// Helper function to check API health
async function checkApiHealth(statusBarItem) {
    try {
        const response = await (0, node_fetch_1.default)(`${API_URL}/health`, {
            method: 'GET',
            signal: AbortSignal.timeout(5000)
        });
        if (response.ok) {
            const data = await response.json();
            statusBarItem.text = `$(zap) Vincent (${data.device})`;
            statusBarItem.tooltip = `Vincent Copilot is active\nModel: ${data.model_id}\nDevice: ${data.device}\n\nPress Tab to accept inline suggestions`;
        }
        else {
            statusBarItem.text = '$(warning) Vincent (API Error)';
            statusBarItem.tooltip = 'Vincent Copilot API is not responding properly';
        }
    }
    catch (error) {
        statusBarItem.text = '$(x) Vincent (Offline)';
        statusBarItem.tooltip = 'Vincent Copilot API is offline. Start the server with: python backend/api_server.py';
    }
}
function deactivate() {
    if (inlineCompletionTimeout) {
        clearTimeout(inlineCompletionTimeout);
    }
}
//# sourceMappingURL=extension.js.map