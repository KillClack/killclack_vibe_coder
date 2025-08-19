import * as vscode from 'vscode'

export class VibeCoderViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'vibeCoder.panel'
  private _view?: vscode.WebviewView
  private _onDidChangeWebview = new vscode.EventEmitter<vscode.WebviewView | undefined>()
  public readonly onDidChangeWebview = this._onDidChangeWebview.event

  constructor(
    private readonly _extensionUri: vscode.Uri,
    private readonly getWebviewContent: () => string,
    private readonly handleMessage: (message: any) => void | Promise<void>
  ) {}

  public resolveWebviewView(
    webviewView: vscode.WebviewView,
    context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken,
  ) {
    this._view = webviewView

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri]
    }

    webviewView.webview.html = this.getWebviewContent()

    webviewView.webview.onDidReceiveMessage(async (message) => {
      await this.handleMessage(message)
    })

    // Notify that the webview is ready
    this._onDidChangeWebview.fire(this._view)

    // Handle visibility changes
    webviewView.onDidChangeVisibility(() => {
      if (webviewView.visible) {
        // Refresh content when becoming visible if needed
        webviewView.webview.html = this.getWebviewContent()
        this._onDidChangeWebview.fire(this._view)
      }
    })

    // Handle disposal
    webviewView.onDidDispose(() => {
      this._view = undefined
      this._onDidChangeWebview.fire(undefined)
    })
  }

  public get webview(): vscode.Webview | undefined {
    return this._view?.webview
  }

  public get visible(): boolean {
    return this._view?.visible ?? false
  }

  public show(preserveFocus?: boolean): void {
    if (this._view) {
      this._view.show?.(preserveFocus)
    }
  }

  public postMessage(message: any): Thenable<boolean> | undefined {
    return this._view?.webview.postMessage(message)
  }

  public dispose() {
    this._view = undefined
  }
}