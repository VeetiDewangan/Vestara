import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import '../../../core/network/api_config.dart';
import 'wardrobe_item.dart';

/// The user's saved wardrobe. Connected to the backend Cloud Database
/// (SQLAlchemy/SQLite) instead of local device storage.
///
/// Widgets listen to it (ListenableBuilder) and rebuild on add/remove.
class WardrobeStore extends ChangeNotifier {
  WardrobeStore();

  /// App-wide instance. Tests create their own with `WardrobeStore()`.
  static final WardrobeStore instance = WardrobeStore();

  final List<WardrobeItem> _items = [];

  /// Newest first.
  List<WardrobeItem> get items => List.unmodifiable(_items);

  String get _itemsUrl => '${ApiConfig.baseUrl}/api/v1/items/';

  Future<void> load() async {
    try {
      final response = await http.get(Uri.parse(_itemsUrl));
      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        _items.clear();
        if (decoded is List) {
          for (final entry in decoded) {
            try {
              _items.add(WardrobeItem.fromJson(entry as Map<String, dynamic>));
            } catch (_) {
              // skip one corrupt entry
            }
          }
          // Sort to ensure newest items appear first
          _items.sort((a, b) => b.createdAt.compareTo(a.createdAt));
        }
      } else {
        debugPrint('Failed to load from cloud: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Could not load wardrobe from cloud: $e');
    }
    notifyListeners();
  }

  Future<void> add(WardrobeItem item) async {
    try {
      final response = await http.post(
        Uri.parse(_itemsUrl),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'filename': item.filename,
          'category': item.category,
          'subcategory': item.subcategory,
          'colors': item.colors,
          'tags': item.tags,
        }),
      );
      if (response.statusCode == 200) {
        final savedItem = WardrobeItem.fromJson(
            jsonDecode(response.body) as Map<String, dynamic>);
        _items.insert(0, savedItem);
        notifyListeners();
      } else {
        debugPrint('Failed to save to cloud: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      debugPrint('Could not save wardrobe to cloud: $e');
    }
  }

  Future<void> remove(String id) async {
    try {
      final response = await http.delete(Uri.parse('$_itemsUrl$id'));
      if (response.statusCode == 200) {
        _items.removeWhere((i) => i.id == id);
        notifyListeners();
      } else {
        debugPrint('Failed to delete from cloud: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Could not delete wardrobe item from cloud: $e');
    }
  }
}
