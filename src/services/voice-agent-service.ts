import * as vscode from 'vscode'
import WebSocket from 'ws'
import { createClient } from '@deepgram/sdk'
// Wrappers & local services
import { MicrophoneWrapper } from '../utils/native-module-wrapper'
import { PromptManagementService } from './prompt-management-service'
import { LLMService } from './llm-service'
import { EventEmitter } from 'events'
import { CommandRegistryService } from './command-registry-service'
import { WorkspaceService } from './workspace-service'
import { ConversationLoggerService } from './conversation-logger-service'
import { SpecGeneratorService } from './spec-generator-service'
import { checkNativeModulesCompatibility } from '../utils/binary-loader'

/**
 * ------------------------------
 * Types for Deepgram Agent V1
 * ------------------------------
 */
interface AgentConfig {
  type: 'Settings'
  audio: {
    input: { encoding: string; sample_rate: number }
    output: { encoding: string; sample_rate: number; container: 'none' | 'wav' }
  }
  agent: {
    language?: string
    greeting?: string
    listen: { provider: { type: string; model: string } }
    think: {
      provider: { type: string; model: string }
      prompt: string
      functions: Array<{
        name: string
        description: string
        parameters: {
          type: 'object'
          properties: {
            name?: { type: string; description: string }
            args?: { type: 'array'; description: string; items: { type: string } }
            format?: { type: string; enum?: string[]; description: string }
            filePath?: { type: string; description: string }
            startLine?: { type: string; description: string }
            endLine?: { type: string; description: string }
          }
          required: string[]
        }
      }>
    }
    speak: { provider: { type: string; model: string } }
  }
}

// Server -> client JSON messages
interface AgentMessageBase { type: string }
interface WelcomeMessage extends AgentMessageBase { type: 'Welcome' }
interface SettingsAppliedMessage extends AgentMessageBase { type: 'SettingsApplied' }
interface ReadyMessage extends AgentMessageBase { type: 'Ready' }
interface SpeechMessage extends AgentMessageBase { type: 'Speech'; text?: string; is_final?: boolean }
interface AgentResponseMessage extends AgentMessageBase {
  type: 'AgentResponse'
  text?: string
  audio?: { data: string; encoding: string; sample_rate: number; container?: string; bitrate?: number }
  tts_latency?: number
  ttt_latency?: number
  total_latency?: number
}
interface ConversationTextMessage extends AgentMessageBase {
  type: 'ConversationText'
  role?: 'assistant' | 'user'
  content?: string
}
interface FunctionCallRequestMessage extends AgentMessageBase {
  type: 'FunctionCallRequest'
  functions?: Array<{ id: string; name: string; arguments: string; client_side: boolean }>
}
interface AgentThinkingMessage extends AgentMessageBase { type: 'AgentThinking'; content?: string }
interface PromptUpdatedMessage extends AgentMessageBase { type: 'PromptUpdated' }
interface SpeakUpdatedMessage extends AgentMessageBase { type: 'SpeakUpdated' }
interface UserStartedSpeakingMessage extends AgentMessageBase { type: 'UserStartedSpeaking' }
interface AgentStartedSpeakingMessage extends AgentMessageBase { type: 'AgentStartedSpeaking' }
interface AgentAudioDoneMessage extends AgentMessageBase { type: 'AgentAudioDone' }
interface WarningMessage extends AgentMessageBase { type: 'Warning'; description?: string }
interface ErrorMessage extends AgentMessageBase { type: 'Error'; description?: string; message?: string; code?: string }
interface CloseMessage extends AgentMessageBase { type: 'Close' }

type AgentMessage =
  | WelcomeMessage
  | SettingsAppliedMessage
  | ReadyMessage
  | SpeechMessage
  | AgentResponseMessage
  | FunctionCallRequestMessage
  | ConversationTextMessage
  | UserStartedSpeakingMessage
  | AgentStartedSpeakingMessage
  | AgentAudioDoneMessage
  | AgentThinkingMessage
  | PromptUpdatedMessage
  | SpeakUpdatedMessage
  | WarningMessage
  | ErrorMessage
  | CloseMessage

export interface MessageHandler {
  postMessage(message: unknown): Thenable<boolean>
}

/**
 * ----------------------------------
 * VoiceAgentService — full rewrite
 * ----------------------------------
 * Key fixes vs. original:
 *  - Properly distinguishes JSON vs. binary frames (no JSON.parse on audio)
 *  - Avoids dumping base64 audio into transcript
 *  - Centralizes KeepAlive (JSON + ws ping)
 *  - Safer microphone lifecycle and cleanup
 *  - Clear logging & event emission
 */
export class VoiceAgentService {
  // Connection
  private ws: WebSocket | null = null
  private keepAliveInterval: NodeJS.Timeout | null = null
  private jsonKeepAliveInterval: NodeJS.Timeout | null = null
  private isInitialized = false

  // Audio / IO
  private readonly INPUT_SAMPLE_RATE = 16000 // mic capture
  private readonly OUTPUT_SAMPLE_RATE = 24000 // agent output
  private readonly OUTPUT_ENCODING = 'linear16' as const
  private currentMic: MicrophoneWrapper | null = null
  private isProcessingRequest = false // Tracks if "Working on that…" has been sent for current request

  // Services
  private promptManager: PromptManagementService
  private llmService: LLMService
  private eventEmitter = new EventEmitter()
  private commandRegistry: CommandRegistryService
  private workspaceService: WorkspaceService
  private agentPanel: MessageHandler | undefined = undefined
  private conversationLogger: ConversationLoggerService
  private specGenerator: SpecGeneratorService

  // VS Code wiring
  private context: vscode.ExtensionContext
  private updateStatus: (status: string) => void
  private updateTranscript: (text: string) => void

  constructor({
    context,
    updateStatus,
    updateTranscript,
    conversationLogger,
  }: {
    context: vscode.ExtensionContext
    updateStatus: (status: string) => void
    updateTranscript: (text: string) => void
    conversationLogger: ConversationLoggerService
  }) {
    this.context = context
    this.updateStatus = updateStatus
    this.updateTranscript = updateTranscript
    this.conversationLogger = conversationLogger

    // Service init order matters here
    this.llmService = new LLMService(context)
    this.specGenerator = new SpecGeneratorService(this.llmService, this.conversationLogger)
    this.promptManager = new PromptManagementService(context)
    this.commandRegistry = new CommandRegistryService()
    this.workspaceService = new WorkspaceService()

    // Register sample function-callable command
    this.commandRegistry.registerCommand({
      name: 'generateProjectSpec',
      command: 'vibe-coder.generateProjectSpec',
      category: 'workspace',
      description: 'Generate a structured project specification from our conversation',
      parameters: {
        type: 'object',
        properties: {
          format: { type: 'string', enum: ['markdown'], description: 'Output format' },
        },
        required: ['format'],
      },
    })

    vscode.window.onDidChangeActiveTextEditor(async () => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        const prompt = await this.getAgentPrompt();
        this.updatePrompt(prompt);
      }
    });
  }

  async initialize(): Promise<void> {
    const compatibility = checkNativeModulesCompatibility();
    if (!compatibility.compatible) {
      vscode.window.showWarningMessage(compatibility.message);
      console.warn('Native module compatibility check failed:', compatibility);
    }
    // Catch unhandled promise rejections
    process.on('unhandledRejection', (reason, promise) => {
      const errMsg = `Unhandled promise rejection: ${reason}`;
      console.error(errMsg);
      this.updateTranscript(errMsg);
      this.conversationLogger.logEntry({ role: 'user', content: errMsg });
    });
    // We don’t require API key at init; we’ll prompt on connect if missing
    this.isInitialized = true;
  }

  /** Connects to Deepgram Agent V1 and wires up event handlers */
  async startAgent(): Promise<void> {
    if (!this.isInitialized) throw new Error('Voice Agent not initialized')

    this.cleanup() // ensure clean slate

    try {
      this.updateStatus('Connecting to agent...')

      let apiKey = await this.context.secrets.get('deepgram.apiKey')
      if (!apiKey) {
        const key = await vscode.window.showInputBox({
          prompt: 'Enter your Deepgram API key',
          password: true,
          placeHolder: 'Deepgram API key is required for voice agent',
          ignoreFocusOut: true,
        })
        if (!key) {
          this.updateStatus('API key required')
          vscode.window.showErrorMessage('Deepgram API key is required for voice agent')
          throw new Error('Deepgram API key is required')
        }
        await this.context.secrets.store('deepgram.apiKey', key)
        apiKey = key
      }

      // Connect to V1 Agent WS endpoint
      this.ws = new WebSocket('wss://agent.deepgram.com/v1/agent/converse', ['token'], {
        headers: { Authorization: `Token ${apiKey}` },
      })

      await new Promise<void>((resolve, reject) => {
        if (!this.ws) return reject(new Error('WebSocket not initialized'))
        this.ws.once('open', () => resolve())
        this.ws.once('error', (err) => reject(err))
      })

      // ---- Message handling (binary vs JSON) ----
      this.ws.on('message', async (data: WebSocket.Data, isBinary?: boolean) => {
        try {
          const binary = typeof isBinary === 'boolean' ? isBinary : this.isBinaryFrame(data)
          if (binary) {
            this.handleRawAudio(this.toBuffer(data))
            return
          }

          const text = typeof data === 'string' ? data : this.toBuffer(data).toString('utf8')
          const message = JSON.parse(text) as AgentMessage
          await this.routeJsonMessage(message)
        } catch (err) {
          // Only warn for unexpected non-JSON text frames; binary is handled above
          console.warn('Ignoring non-JSON text frame:', (err as Error).message)
        }
      })

      // Keep-alives: TCP/WebSocket ping and Agent-level KeepAlive JSON
      this.keepAliveInterval = setInterval(() => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.ping()
      }, 30_000)

      this.jsonKeepAliveInterval = setInterval(() => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: 'KeepAlive' }))
        }
      }, 25_000)

      this.updateStatus('Connected, awaiting Welcome...')
    } catch (error) {
      console.error('Failed to start agent:', error)
      this.cleanup()
      throw error
    }
  }

  /** Properly routes parsed JSON messages */
  private async routeJsonMessage(message: AgentMessage): Promise<void> {
    switch (message.type) {
      case 'Welcome': {
        this.updateStatus('Sending configuration...')
        const config = await this.getAgentConfig()
        this.ws?.send(JSON.stringify(config))
        break
      }
      case 'SettingsApplied': {
        this.updateStatus('Settings applied, starting microphone...')
        this.setupMicrophone()
        break
      }
      case 'Ready': {
        this.updateStatus('Ready! Start speaking...')
        break
      }
      case 'Speech': {
        const s = (message as SpeechMessage).text || ''
        if (s) this.updateTranscript(s)
        break
      }
      case 'AgentResponse': {
        const m = message as AgentResponseMessage
        if (m.text) {
          this.conversationLogger.logEntry({ role: 'assistant', content: m.text })
          this.updateTranscript(m.text)
          this.isProcessingRequest = false; // Reset after agent responds
        }
        // Some servers include base64 audio in JSON; still support it.
        if (m.audio && this.agentPanel) {
          this.agentPanel.postMessage({
            type: 'playAudio',
            audio: { data: m.audio.data, encoding: m.audio.encoding, sampleRate: m.audio.sample_rate },
          })
          this.sendSpeakingStateUpdate('speaking')
        }
        break
      }
      case 'FunctionCallRequest': {
        const fmsg = message as FunctionCallRequestMessage
        const funcs = fmsg.functions || []
        for (const func of funcs) {
          if (!func.client_side) continue
          try {
            const result = await this.handleFunctionCall(func.id, { name: func.name, arguments: func.arguments })
            const response = { type: 'FunctionCallResponse', id: func.id, name: func.name, content: JSON.stringify(result) }
            this.ws?.send(JSON.stringify(response))
          } catch (e) {
            const errRes = {
              type: 'FunctionCallResponse',
              id: func.id,
              name: func.name,
              content: JSON.stringify({ success: false, error: (e as Error).message }),
            }
            this.ws?.send(JSON.stringify(errRes))
          }
        }
        break
      }
      case 'ConversationText': {
        const c = message as ConversationTextMessage
        if (c.role && c.content) {
          this.conversationLogger.logEntry({ role: c.role, content: c.content })
          if (c.role === 'assistant') {
            this.agentPanel?.postMessage({
              type: 'updateTranscript',
              text: c.content,
              target: 'agent-transcript',
              animate: true,
            })
          }
          this.eventEmitter.emit('transcript', c.content)
        }
        break
      }
      case 'UserStartedSpeaking': {
        // Stop any active playback in the webview
        this.agentPanel?.postMessage({ type: 'stopAudio' })
        this.sendSpeakingStateUpdate('idle')
        this.isProcessingRequest = false; // Reset for new user request
        break
      }
      case 'AgentStartedSpeaking': {
        this.sendSpeakingStateUpdate('speaking')
        break
      }
      case 'AgentAudioDone': {
        this.sendSpeakingStateUpdate('idle')
        break
      }
      case 'AgentThinking': {
        this.updateStatus('Agent thinking...')
        break
      }
      case 'PromptUpdated':
      case 'SpeakUpdated': {
        // Informational
        break
      }
      case 'Warning': {
        const w = message as WarningMessage
        vscode.window.showWarningMessage(`Agent warning: ${w.description ?? 'Unknown warning'}`)
        break
      }
      case 'Error': {
        const e = message as ErrorMessage
        vscode.window.showErrorMessage(`Agent error: ${e.description || e.message || 'Unknown error'}`)
        this.updateStatus('Error occurred')
        break
      }
      case 'Close': {
        this.cleanup()
        break
      }
      default: {
        console.log('Unknown message type:', (message as any)?.type)
      }
    }
  }

  /** Microphone -> stream PCM16 to WS */
  private setupMicrophone() {
    // Ensure previous mic is stopped
    if (this.currentMic) {
      try { this.currentMic.stopRecording() } catch {}
      this.currentMic = null
    }

    const mic = new MicrophoneWrapper()
    this.currentMic = mic

    try {
      const audioStream = mic.startRecording()

      audioStream.on('data', (chunk: Buffer) => {
        if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(chunk)
      })
      audioStream.on('error', (error: Error) => {
        vscode.window.showErrorMessage(`Microphone error: ${error.message}`)
        this.cleanup()
      })
      this.updateStatus('Mic streaming… speak anytime')
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to start microphone: ${error instanceof Error ? error.message : String(error)}`)
      this.cleanup()
    }
  }

  /** Binary audio from server (raw linear16 or WAV) */
  private handleRawAudio(data: Buffer) {
    if (!this.agentPanel) return
    const base64Audio = data.toString('base64')
    this.agentPanel.postMessage({
      type: 'playAudio',
      audio: { data: base64Audio, encoding: this.OUTPUT_ENCODING, sampleRate: this.OUTPUT_SAMPLE_RATE, isRaw: true },
    })
    this.sendSpeakingStateUpdate('speaking')
    setTimeout(() => this.sendSpeakingStateUpdate('idle'), 1000)
  }

  /** Update the system prompt at runtime */
  async updatePrompt(prompt: string) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new Error('Agent not connected')
    const updateMessage = { type: 'UpdatePrompt', prompt }
    this.ws.send(JSON.stringify(updateMessage))
  }

  /** Switch TTS model at runtime */
  async updateSpeakModel(model: string) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new Error('Agent not connected')
    const updateMessage = { type: 'UpdateSpeak', speak: { provider: { type: 'deepgram', model } } }
    this.ws.send(JSON.stringify(updateMessage))
  }

  /** Inject a message spoken by the agent */
  async injectAgentMessage(message: string) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new Error('Agent not connected')
    const injectMessage = { type: 'InjectAgentMessage', content: message }
    this.ws.send(JSON.stringify(injectMessage))
  }

  /** Execute function call requests from the agent */
  private async handleFunctionCall(functionCallId: string, func: { name: string; arguments: string }) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new Error('Agent not connected')

    try {
      if (!this.isProcessingRequest) {
        await this.injectAgentMessage('Working on that…');
        this.isProcessingRequest = true;
      }
    } catch {}

    if (func.name === 'generateProjectSpec') {
      try {
        await this.specGenerator.generateSpec()
        const msg = 'Project specification has been generated and saved to project_spec.md'
        this.updateTranscript(msg)
        this.conversationLogger.logEntry({ role: 'assistant', content: msg })
        return { success: true, message: msg }
      } catch (error) {
        const errMsg = `Failed to generate project specification: ${(error as Error).message}`
        this.updateTranscript(errMsg)
        this.conversationLogger.logEntry({ role: 'assistant', content: errMsg })
        return { success: false, error: errMsg }
      }
    }

    if (func.name === 'execute_command') {
      const args = JSON.parse(func.arguments || '{}') as { name: string; args?: string[] }
      try {
        // Check if the command exists
        const availableCommands = await vscode.commands.getCommands(true);
        if (!availableCommands.includes(args.name)) {
          const errMsg = `Command '${args.name}' not found in VS Code.`;
          this.updateTranscript(errMsg);
          this.conversationLogger.logEntry({ role: 'assistant', content: errMsg });
          return { success: false, error: errMsg };
        }
        await this.commandRegistry.executeCommand(args.name, args.args);
        return { success: true };
      } catch (error) {
        const errMsg = `Failed to execute command '${args.name}': ${(error as Error).message}`;
        this.updateTranscript(errMsg);
        this.conversationLogger.logEntry({ role: 'assistant', content: errMsg });
        return { success: false, error: errMsg };
      }
    }

    if (func.name === 'readFile') {
      const args = JSON.parse(func.arguments || '{}') as { filePath: string; startLine?: number; endLine?: number }
      try {
        const fileUri = vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0].uri, args.filePath)
        const content = await vscode.workspace.fs.readFile(fileUri)
        let result = content.toString()
        if (args.startLine && args.endLine) {
          const lines = result.split('\n')
          const start = Math.max(1, args.startLine) - 1
          const end = Math.min(lines.length, args.endLine)
          result = lines.slice(start, end).join('\n')
        }
        const msg = `File ${args.filePath} read successfully${args.startLine && args.endLine ? ` (lines ${args.startLine}-${args.endLine})` : ''}`
        this.updateTranscript(msg)
        this.conversationLogger.logEntry({ role: 'assistant', content: msg })
        return { success: true, content: result, message: msg }
      } catch (error) {
        const errMsg = `Failed to read file ${args.filePath}: ${(error as Error).message}`
        this.updateTranscript(errMsg)
        this.conversationLogger.logEntry({ role: 'assistant', content: errMsg })
        return { success: false, error: errMsg }
      }
    }

    if (func.name === 'navigateToFile') {
      const args = JSON.parse(func.arguments || '{}') as { filePath: string; startLine?: number; endLine?: number }
      try {
        const fileUri = vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0].uri, args.filePath)
        const document = await vscode.workspace.openTextDocument(fileUri)
        const editor = await vscode.window.showTextDocument(document)
        if (args.startLine) {
          const start = new vscode.Position(Math.max(0, args.startLine - 1), 0)
          const end = args.endLine ? new vscode.Position(Math.max(0, args.endLine - 1), 0) : start
          editor.selection = new vscode.Selection(start, end)
          editor.revealRange(new vscode.Range(start, end))
        }
        const msg = `Navigated to file ${args.filePath}${args.startLine ? ` at line ${args.startLine}` : ''}`
        this.updateTranscript(msg)
        this.conversationLogger.logEntry({ role: 'assistant', content: msg })
        this.isProcessingRequest = false; // Reset after successful navigation
        return { success: true, message: msg }
      } catch (error) {
        const errMsg = `Failed to navigate to file ${args.filePath}: ${(error as Error).message}`
        this.updateTranscript(errMsg)
        this.conversationLogger.logEntry({ role: 'assistant', content: errMsg })
        return { success: false, error: errMsg }
      }
    }

    return { success: false, error: `Unknown function: ${func.name}` }
  }

    /** Build Settings message sent after Welcome */
  private async getAgentConfig(): Promise<AgentConfig> {
    const commands = this.commandRegistry.getCommandDefinitions();
    const prompt = await this.getAgentPrompt();

    return {
      type: 'Settings',
      audio: {
        input: { encoding: 'linear16', sample_rate: this.INPUT_SAMPLE_RATE },
        output: { encoding: 'linear16', sample_rate: this.OUTPUT_SAMPLE_RATE, container: 'none' },
      },
      agent: {
        language: 'en',
        listen: { provider: { type: 'deepgram', model: 'nova-3' } },
        think: {
          provider: { type: 'open_ai', model: 'gpt-4.1-mini' },
          prompt,
          functions: [
            {
              name: 'execute_command',
              description: 'Execute a VS Code command',
              parameters: {
                type: 'object',
                properties: {
                  name: { type: 'string', description: 'Name of the command to execute' },
                  args: { type: 'array', description: 'Arguments for the command', items: { type: 'string' } },
                },
                required: ['name'],
              },
            },
            {
              name: 'generateProjectSpec',
              description: 'Generate a project specification document from the conversation history',
              parameters: {
                type: 'object',
                properties: { format: { type: 'string', enum: ['markdown'], description: 'Output format' } },
                required: ['format'],
              },
            },
            {
              name: 'readFile',
              description: 'Read the contents of a specific file, optionally from a start line to an end line',
              parameters: {
                type: 'object',
                properties: {
                  filePath: { type: 'string', description: 'Relative path to the file in the workspace' },
                  startLine: { type: 'number', description: 'Starting line number (1-based, optional)' },
                  endLine: { type: 'number', description: 'Ending line number (1-based, optional)' },
                },
                required: ['filePath'],
              },
            },
            {
              name: 'navigateToFile',
              description: 'Navigate to a specific file in the VS Code editor, optionally to a specific line range',
              parameters: {
                type: 'object',
                properties: {
                  filePath: { type: 'string', description: 'Relative path to the file in the workspace' },
                  startLine: { type: 'number', description: 'Starting line number to navigate to (1-based, optional)' },
                  endLine: { type: 'number', description: 'Ending line number to select (1-based, optional)' },
                },
                required: ['filePath'],
              },
            },
          ],
        },
        speak: { provider: { type: 'deepgram', model: 'aura-2-thalia-en' } },
      },
    };
  }

  /** Generate the agent prompt with workspace and active file context */
  private async getAgentPrompt(): Promise<string> {
      const fileTree = await this.workspaceService.getFileTree();
      const formattedTree = this.workspaceService.formatFileTree(fileTree);
      return `You are a coding mentor and VS Code assistant. You help users navigate and control VS Code through voice commands. Ask questions one at a time using the Socratic method unless explicitly asked for suggestions.
        Everything you say will be spoken aloud via TTS, so avoid formatting and keep responses concise.

        Be proactive in assisting users. For tasks like finding a specific part of code or navigating to it, actively search by calling readFile on relevant files (using line ranges if possible to be efficient), analyze the returned content, and if not found, call readFile on other potential files. Once located, use navigateToFile to move the editor there. Do not speak or respond until the task is completed (e.g., navigation done) or if you need clarification from the user. You can make multiple function calls in sequence or parallel as needed before concluding and speaking.

        When calling execute_command, ensure the command is valid for VS Code (e.g., 'workbench.action.openEditorAtIndex', 'workbench.action.gotoLine'). If unsure about a command's existence, avoid calling it and instead inform the user that the command is not recognized, suggesting they clarify or provide a different command.

        Current Workspace Structure:
        ${formattedTree}

        Current Active File:
        ${this.getActiveFileContext()}

        When a user requests an action matching a VS Code command, call execute_command only if the command is valid.
        You can help users navigate the file structure and open files using the paths shown above.
        You can also generate project specifications from our conversation using the generateProjectSpec function.
        You can read file contents using the readFile function, optionally specifying line ranges.
        You can navigate to specific files and line ranges using the navigateToFile function.
        Provide helpful feedback about what you're doing and guide users if they need help.`;
  }
  /** Send a text message to the agent and await first AgentResponse */
  private async sendToAgent(text: string) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new Error('Agent not connected')

    const message = { type: 'UserText', text }
    this.ws.send(JSON.stringify(message))

    return new Promise<{ text?: string }>((resolve) => {
      const handler = (data: WebSocket.Data, isBinary?: boolean) => {
        if (isBinary || this.isBinaryFrame(data)) return // ignore audio frames here
        try {
          const json = JSON.parse(typeof data === 'string' ? data : this.toBuffer(data).toString('utf8'))
          if (json?.type === 'AgentResponse') {
            this.ws?.off('message', handler as any)
            resolve({ text: json.text })
          }
        } catch {}
      }
      this.ws?.on('message', handler as any)
    })
  }

  /** Public API: set webview bridge */
  public setAgentPanel(handler: MessageHandler | undefined) {
    this.agentPanel = handler
  }

  /** Event subscription for transcripts */
  onTranscript(callback: (text: string) => void) {
    this.eventEmitter.on('transcript', callback)
    return () => this.eventEmitter.off('transcript', callback)
  }

  /** Update webview UI state */
  private sendSpeakingStateUpdate(state: 'speaking' | 'idle') {
    if (!this.agentPanel) return
    this.agentPanel.postMessage({
      type: 'updateStatus',
      text: state === 'speaking' ? 'Agent Speaking...' : 'Ready',
      target: 'vibe-status',
    })
  }

  /** Dispose all resources */
  public dispose(): void { this.cleanup() }

  /** Disconnect and clear timers/streams */
  public cleanup(): void {
    console.log('Cleaning up voice agent...')

    if (this.currentMic) {
      try { this.currentMic.stopRecording() } catch {}
      this.currentMic = null
    }

    if (this.ws) {
      try { this.ws.removeAllListeners() } catch {}
      try { this.ws.close() } catch {}
      this.ws = null
    }
    if (this.keepAliveInterval) { clearInterval(this.keepAliveInterval); this.keepAliveInterval = null }
    if (this.jsonKeepAliveInterval) { clearInterval(this.jsonKeepAliveInterval); this.jsonKeepAliveInterval = null }

    this.updateStatus('Disconnected')
  }

  /** Helper: detect binary frame shapes from ws */
  private isBinaryFrame(d: WebSocket.Data): boolean {
    return Buffer.isBuffer(d) || d instanceof ArrayBuffer || Array.isArray(d)
  }

  /** Helper: normalize to Buffer */
  private toBuffer(d: WebSocket.Data): Buffer {
    if (Buffer.isBuffer(d)) return d
    if (Array.isArray(d)) return Buffer.concat(d as Buffer[])
    if (d instanceof ArrayBuffer) return Buffer.from(new Uint8Array(d))
    return Buffer.from(d as any)
  }

  /** Get context of the currently active file */
  private getActiveFileContext(): string {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) return "No active file.";
    const filePath = vscode.workspace.asRelativePath(activeEditor.document.uri.fsPath);
    const content = activeEditor.document.getText();
    return `File: ${filePath}\nContent:\n${content.slice(0, 1000)}${content.length > 1000 ? '...' : ''}`;
  }

  /** Example of handling an incoming message from your own UI */
  private async handleMessage(message: any) {
    if (message.type === 'text') {
      this.conversationLogger.logEntry({ role: 'user', content: message.text })
      const response = await this.sendToAgent(message.text)
      if (response.text) this.conversationLogger.logEntry({ role: 'assistant', content: response.text })
      this.updateTranscript(response.text || 'No response from agent')
    }
  }
}