FROM python:3.9

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# setup github token for private repo access
RUN --mount=type=secret,id=_env,dst=/etc/secrets/.env .
/etc/secrets/.env \
&& echo "machine github.com login __token__ password
${INSTAFS_TOKEN}" >> /root/.netrc \
&& chmod 600 /root/.netrc
WORKDIR /app
COPY megascale/ megascale/
# TODO copy other necessary files
# TODO install dependencies
# Add uv Python to PATH
ENV PATH="/app/.venv/bin:$PATH"
COPY config/ config/
CMD python main.py