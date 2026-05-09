if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

if [ -f "$HOME/.elan/env" ]; then
    . "$HOME/.elan/env"
fi

export PATH="$HOME/.elan/bin:$PATH"
export PATH="$HOME/.local/bin:$PATH"

case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
