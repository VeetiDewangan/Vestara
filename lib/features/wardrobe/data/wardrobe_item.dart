import 'dart:convert';
import 'dart:typed_data';

import '../../../core/network/api_config.dart';

/// One saved clothing item.
///
/// The cut-out picture itself is saved on the device ([imageData], a small
/// base64 image), so the item shows even if the backend is off or its address
/// changed (e.g. emulator -> real phone). Older items without [imageData]
/// fall back to downloading [filename] from the backend.
class WardrobeItem {
  const WardrobeItem({
    required this.id,
    required this.filename,
    required this.category,
    required this.subcategory,
    required this.tags,
    required this.colors,
    required this.createdAt,
    this.imageData,
  });

  final String id;
  final String filename;
  final String category; // Top / Bottom / Outerwear / One-piece
  final String subcategory; // T-Shirt, Jeans, ...
  final List<String> tags;
  final List<String> colors;
  final DateTime createdAt;

  /// Base64 of the small transparent cut-out kept on the device (may be null
  /// for items saved by older versions of the app).
  final String? imageData;

  static final Expando<Uint8List> _decoded = Expando<Uint8List>('imageBytes');

  /// Decoded [imageData] (decoded once, then reused on every rebuild).
  Uint8List? get imageBytes {
    final data = imageData;
    if (data == null || data.isEmpty) return null;
    final cached = _decoded[this];
    if (cached != null) return cached;
    try {
      final bytes = base64Decode(data);
      _decoded[this] = bytes;
      return bytes;
    } on FormatException {
      return null;
    }
  }

  /// Transparent PNG cut-out served by the backend.
  String get imageUrl =>
      '${ApiConfig.baseUrl}${ApiConfig.clothingPath}/result/$filename';

  /// e.g. "Navy T-Shirt"
  String get displayName {
    if (colors.isEmpty) return subcategory;
    final c = colors.first;
    return '${c[0].toUpperCase()}${c.substring(1)} $subcategory';
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'filename': filename,
        'category': category,
        'subcategory': subcategory,
        'tags': tags,
        'colors': colors,
        'created_at': createdAt.toIso8601String(),
        if (imageData != null) 'image_data': imageData,
      };

  factory WardrobeItem.fromJson(Map<String, dynamic> json) {
    List<String> strings(dynamic v) =>
        v is List ? v.whereType<String>().toList() : <String>[];
    return WardrobeItem(
      id: json['id'] as String,
      filename: json['filename'] as String,
      category: (json['category'] as String?) ?? 'Top',
      subcategory: (json['subcategory'] as String?) ?? 'Top',
      tags: strings(json['tags']),
      colors: strings(json['colors']),
      createdAt:
          DateTime.tryParse((json['created_at'] as String?) ?? '') ?? DateTime.now(),
      imageData: json['image_data'] as String?,
    );
  }
}
