import 'dart:convert';
import 'dart:typed_data';

/// Parsed response of `POST /api/v1/clothing/analyze`.
///
/// NOTE: when AI tagging is unavailable the backend still succeeds but sends
/// `null` for category / pattern / material and empty lists for the rest, so
/// every field here is nullable-safe.

List<String> _stringList(dynamic value) {
  if (value is List) {
    return value
        .whereType<String>()
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();
  }
  return const [];
}

String? _nullableString(dynamic value) {
  if (value is String && value.trim().isNotEmpty) return value.trim();
  return null;
}

class ClothingTags {
  const ClothingTags({
    this.category,
    this.colors = const [],
    this.pattern,
    this.style = const [],
    this.season = const [],
    this.occasions = const [],
    this.fit,
    this.neckline,
    this.materialGuess,
    this.extraTags = const [],
  });

  final String? category;
  final List<String> colors;
  final String? pattern;
  final List<String> style;
  final List<String> season;

  /// Occasions this garment is suitable for (e.g. "wedding", "daily wear").
  final List<String> occasions;

  /// Silhouette / fit type (e.g. "a-line", "slim fit", "oversized").
  final String? fit;

  /// Neckline type (e.g. "v-neck", "off-shoulder", "sweetheart").
  final String? neckline;

  final String? materialGuess;
  final List<String> extraTags;

  factory ClothingTags.fromJson(Map<String, dynamic> json) => ClothingTags(
        category: _nullableString(json['category']),
        colors: _stringList(json['colors']),
        pattern: _nullableString(json['pattern']),
        style: _stringList(json['style']),
        season: _stringList(json['season']),
        occasions: _stringList(json['occasions']),
        fit: _nullableString(json['fit']),
        neckline: _nullableString(json['neckline']),
        materialGuess: _nullableString(json['material_guess']),
        extraTags: _stringList(json['extra_tags']),
      );
}

class TaggingInfo {
  const TaggingInfo({required this.source, this.model, this.error});

  /// "gemini" when the AI produced the tags, "local" for the colors-only fallback.
  final String source;
  final String? model;
  final String? error;

  bool get isAi => source == 'gemini';

  factory TaggingInfo.fromJson(Map<String, dynamic> json) => TaggingInfo(
        source: (json['source'] as String?) ?? 'local',
        model: _nullableString(json['model']),
        error: _nullableString(json['error']),
      );
}

class ClothingAnalysis {
  const ClothingAnalysis({
    required this.filename,
    required this.imageUrl,
    this.imageBase64,
    this.imageBytes,
    required this.width,
    required this.height,
    required this.processingTimeS,
    required this.tags,
    required this.tagList,
    required this.tagging,
  });

  /// Output PNG name on the server (store this if you want to re-fetch later).
  final String filename;

  /// Absolute URL of the transparent PNG cut-out.
  final String imageUrl;

  /// Small transparent copy of the cut-out that the backend puts INSIDE the
  /// reply (base64 text + the decoded bytes). Showing and saving this means
  /// the picture never depends on a second request to [imageUrl].
  final String? imageBase64;
  final Uint8List? imageBytes;
  final int width;
  final int height;
  final double processingTimeS;
  final ClothingTags tags;

  /// Flat, display-ready list for chips (category, colors, pattern, occasions...).
  final List<String> tagList;
  final TaggingInfo tagging;

  /// [baseUrl] is prepended because the backend returns a relative `image_url`.
  factory ClothingAnalysis.fromJson(
    Map<String, dynamic> json, {
    required String baseUrl,
  }) {
    final rel = (json['image_url'] as String?) ?? '';
    final absolute = rel.startsWith('http') ? rel : '$baseUrl$rel';
    final b64 = _nullableString(json['image_base64']);
    Uint8List? bytes;
    if (b64 != null) {
      try {
        bytes = base64Decode(b64);
      } on FormatException {
        bytes = null; // fall back to imageUrl
      }
    }
    return ClothingAnalysis(
      filename: (json['filename'] as String?) ?? '',
      imageUrl: absolute,
      imageBase64: bytes == null ? null : b64,
      imageBytes: bytes,
      width: (json['width'] as num?)?.toInt() ?? 0,
      height: (json['height'] as num?)?.toInt() ?? 0,
      processingTimeS: (json['processing_time_s'] as num?)?.toDouble() ?? 0,
      tags: ClothingTags.fromJson(
        (json['tags'] as Map<String, dynamic>?) ?? const {},
      ),
      tagList: _stringList(json['tag_list']),
      tagging: TaggingInfo.fromJson(
        (json['tagging'] as Map<String, dynamic>?) ?? const {},
      ),
    );
  }

  /// Returns a copy with some fields replaced (e.g. after the user edits the
  /// chips, or after a "try AI tagging again").
  ClothingAnalysis copyWith({
    List<String>? tagList,
    ClothingTags? tags,
    TaggingInfo? tagging,
  }) =>
      ClothingAnalysis(
        filename: filename,
        imageUrl: imageUrl,
        imageBase64: imageBase64,
        imageBytes: imageBytes,
        width: width,
        height: height,
        processingTimeS: processingTimeS,
        tags: tags ?? this.tags,
        tagList: tagList ?? this.tagList,
        tagging: tagging ?? this.tagging,
      );
}
