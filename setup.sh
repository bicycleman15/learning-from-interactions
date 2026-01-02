conda create --name lean python=3.10 -y

# https://github.com/leanprover/elan
# Mac command below
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh

# Set your GitHub access token (don't commit the actual token!)
# export GITHUB_ACCESS_TOKEN="your_token_here"

pip install lean-dojo

pip install openai anthropic