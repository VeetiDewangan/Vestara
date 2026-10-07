import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

import '../../../core/network/api_config.dart';
import 'clothing_analysis.dart';

/// A problem talking to the backend, with a message safe to show to the user.
class ClothingApiException implements Exception {
  ClothingApiException(this.message, {this.statusCode});

  final String message;
  final int? statusCode;

  @override
  String toString() => message;
}

class ClothingApi {
  ClothingApi({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;

  /// Uploads a photo to `POST /analyze`: removes the background AND tags it.
  Future<ClothingAnalysis> analyze(XFile image) async =>
      analyzeBytes(await image.readAsBytes(), image.name);

  /// Same as [analyze] when you already have the bytes (e.g. for a preview).
  Future<ClothingAnalysis> analyzeBytes(
    Uint8List bytes,
    String filename,
  ) async {
    final uri =
        Uri.parse('${ApiConfig.baseUrl}${ApiConfig.clothingPath}/analyze');

    // The backend detects the real format from the bytes, so the filename
    // is only cosmetic. The form field MUST be called "file".
    final request = http.MultipartRequest('POST', uri)
      ..files.add(http.MultipartFile.fromBytes(
        'file',
        bytes,
        filename: filename.isEmpty ? 'upload.jpg' : filename,
      ));

    final http.Response response;
    try {
      final streamed =
          await _client.send(request).timeout(ApiConfig.analyzeTimeout);
      response = await http.Response.fromStream(streamed);
    } on TimeoutException {
      throw ClothingApiException(
        'The server took too long to respond. Please try again.',
      );
    } on http.ClientException {
      throw ClothingApiException(
        "Couldn't reach the server at ${ApiConfig.baseUrl}. "
        'Check that the backend is running and the address is correct.',
      );
    }

    if (response.statusCode != 200) {
      throw ClothingApiException(
        _errorDetail(response),
        statusCode: response.statusCode,
      );
    }

    try {
      final decoded = jsonDecode(utf8.decode(response.bodyBytes));
      return ClothingAnalysis.fromJson(
        decoded as Map<String, dynamic>,
        baseUrl: ApiConfig.baseUrl,
      );
    } on FormatException {
      throw ClothingApiException('The server sent an unexpected reply.');
    } on TypeError {
      throw ClothingApiException('The server sent an unexpected reply.');
    }
  }

  /// Runs ONLY the AI tagging again for a cut-out the backend already made
  /// (`POST /retag/{filename}`), e.g. after a temporary quota / network
  /// problem. Returns [previous] with the new tags; the picture is untouched.
  Future<ClothingAnalysis> retag(ClothingAnalysis previous) async {
    final uri = Uri.parse(
      '${ApiConfig.baseUrl}${ApiConfig.clothingPath}/retag/'
      '${Uri.encodeComponent(previous.filename)}',
    );

    final http.Response response;
    try {
      response = await _client.post(uri).timeout(ApiConfig.analyzeTimeout);
    } on TimeoutException {
      throw ClothingApiException(
        'The server took too long to respond. Please try again.',
      );
    } on http.ClientException {
      throw ClothingApiException(
        "Couldn't reach the server at ${ApiConfig.baseUrl}. "
        'Check that the backend is running and the address is correct.',
      );
    }

    if (response.statusCode != 200) {
      throw ClothingApiException(
        _errorDetail(response),
        statusCode: response.statusCode,
      );
    }

    try {
      final decoded =
          jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
      return previous.copyWith(
        tags: ClothingTags.fromJson(
          (decoded['tags'] as Map<String, dynamic>?) ?? const {},
        ),
        tagList: _stringsOf(decoded['tag_list']),
        tagging: TaggingInfo.fromJson(
          (decoded['tagging'] as Map<String, dynamic>?) ?? const {},
        ),
      );
    } on FormatException {
      throw ClothingApiException('The server sent an unexpected reply.');
    } on TypeError {
      throw ClothingApiException('The server sent an unexpected reply.');
    }
  }

  static List<String> _stringsOf(dynamic value) => value is List
      ? value.whereType<String>().where((s) => s.trim().isNotEmpty).toList()
      : <String>[];

  /// FastAPI errors look like {"detail": "message"} (400, 413, 500).
  String _errorDetail(http.Response response) {
    try {
      final body = jsonDecode(utf8.decode(response.bodyBytes));
      if (body is Map && body['detail'] is String) {
        return body['detail'] as String;
      }
    } catch (_) {
      // fall through to generic messages
    }
    if (response.statusCode == 413) {
      return 'That photo is too large (10 MB max).';
    }
    return 'Something went wrong (HTTP ${response.statusCode}). Please try again.';
  }

  void dispose() => _client.close();
}
