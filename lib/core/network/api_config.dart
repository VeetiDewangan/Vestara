import 'package:flutter/foundation.dart';

/// Where the FastAPI backend (the `backend/` folder in this repo) lives.
///
/// Defaults (work with `python run.py` on the same computer):
///   * Android emulator -> http://10.0.2.2:8000   (10.0.2.2 = your computer)
///   * iOS simulator / desktop -> http://127.0.0.1:8000
///   * Flutter WEB -> the same host the page was opened from, port 8000.
///     (Open the app at http://192.168.1.23:5000 on your phone and it talks to
///     http://192.168.1.23:8000 automatically. "127.0.0.1" would be the PHONE.)
///
/// A REAL phone must use your computer's Wi-Fi IP instead. Pass it at run time:
///   flutter run --dart-define=API_BASE_URL=http://192.168.1.23:8000
/// (and start the backend with HOST=0.0.0.0 so it accepts Wi-Fi connections).
class ApiConfig {
  ApiConfig._();

  static const String _override = String.fromEnvironment('API_BASE_URL');

  static String get baseUrl {
    if (_override.isNotEmpty) return _override;
    if (kIsWeb) {
      final host = Uri.base.host;
      if (host.isNotEmpty) {
        final scheme = Uri.base.scheme == 'https' ? 'https' : 'http';
        return '$scheme://$host:8000';
      }
    }
    if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:8000';
    }
    return 'http://127.0.0.1:8000';
  }

  /// All clothing endpoints are mounted under this prefix by the backend.
  static const String clothingPath = '/api/v1/clothing';

  /// Background removal takes a few seconds; the very first request can take
  /// much longer because the backend downloads the U2-Net model (~176 MB).
  static const Duration analyzeTimeout = Duration(seconds: 180);
}
