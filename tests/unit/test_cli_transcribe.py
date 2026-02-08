"""Testes dos comandos `theo transcribe` e `theo translate`."""

from __future__ import annotations

from click.testing import CliRunner

from theo.cli import cli


class TestTranscribeCommand:
    def test_transcribe_command_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "Transcreve um arquivo de audio" in result.output

    def test_transcribe_requires_model(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "audio.wav"])
        assert result.exit_code != 0
        assert "model" in result.output.lower() or "missing" in result.output.lower()

    def test_transcribe_help_shows_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert "--model" in result.output
        assert "--format" in result.output
        assert "--language" in result.output
        assert "--server" in result.output

    def test_transcribe_format_choices(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert "json" in result.output
        assert "verbose_json" in result.output
        assert "text" in result.output
        assert "srt" in result.output
        assert "vtt" in result.output

    def test_transcribe_file_not_found(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli, ["transcribe", "/tmp/inexistente.wav", "--model", "faster-whisper-tiny"]
        )
        assert result.exit_code != 0
        assert (
            "nao encontrado" in result.output.lower() or "nao disponivel" in result.output.lower()
        )

    def test_transcribe_server_not_running(self, tmp_path: object) -> None:
        # Cria arquivo fake para passar a validacao de arquivo
        import pathlib

        audio_file = pathlib.Path(str(tmp_path)) / "audio.wav"
        audio_file.write_bytes(b"fake audio data")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(audio_file),
                "--model",
                "faster-whisper-tiny",
                "--server",
                "http://localhost:59999",
            ],
        )
        assert result.exit_code != 0
        assert "theo serve" in result.output.lower() or "nao disponivel" in result.output.lower()


class TestTranslateCommand:
    def test_translate_command_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["translate", "--help"])
        assert result.exit_code == 0
        assert "Traduz um arquivo de audio" in result.output

    def test_translate_requires_model(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["translate", "audio.wav"])
        assert result.exit_code != 0

    def test_translate_help_shows_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["translate", "--help"])
        assert "--model" in result.output
        assert "--format" in result.output
        assert "--server" in result.output
        # translate nao tem --language (sempre traduz para ingles)
        assert "--language" not in result.output
