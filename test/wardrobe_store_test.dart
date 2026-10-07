import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:wardrobe_app/features/wardrobe/data/clothing_taxonomy.dart';
import 'package:wardrobe_app/features/wardrobe/data/wardrobe_item.dart';
import 'package:wardrobe_app/features/wardrobe/data/wardrobe_store.dart';

WardrobeItem _item(String id, {List<String> colors = const ['navy']}) => WardrobeItem(
      id: id,
      filename: '$id.png',
      category: 'Top',
      subcategory: 'T-Shirt',
      tags: const ['t-shirt', 'navy'],
      colors: colors,
      createdAt: DateTime(2026, 1, 1),
    );

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() => SharedPreferences.setMockInitialValues({}));

  test('items survive an app restart (save, then load in a new store)', () async {
    final first = WardrobeStore();
    await first.load();
    await first.add(_item('a'));
    await first.add(_item('b'));

    final afterRestart = WardrobeStore();
    await afterRestart.load();

    expect(afterRestart.items.map((i) => i.id), ['b', 'a']); // newest first
    expect(afterRestart.items.first.tags, contains('navy'));
  });

  test('remove deletes and persists', () async {
    final store = WardrobeStore();
    await store.add(_item('a'));
    await store.remove('a');

    final reloaded = WardrobeStore();
    await reloaded.load();
    expect(reloaded.items, isEmpty);
  });

  test('corrupt saved data does not crash the app', () async {
    SharedPreferences.setMockInitialValues({'wardrobe_items_v1': '{not json'});
    final store = WardrobeStore();
    await store.load();
    expect(store.items, isEmpty);
  });

  test('one bad entry is skipped, good entries are kept', () async {
    SharedPreferences.setMockInitialValues({
      'wardrobe_items_v1':
          '[{"oops": true}, {"id":"x","filename":"x.png","category":"Top","subcategory":"Shirt"}]',
    });
    final store = WardrobeStore();
    await store.load();
    expect(store.items.map((i) => i.id), ['x']);
  });

  test('display name uses the first color', () {
    expect(_item('a').displayName, 'Navy T-Shirt');
    expect(_item('a', colors: const []).displayName, 'T-Shirt');
  });

  test('every AI category maps to a subcategory the UI can show', () {
    const aiCategories = [
      't-shirt', 'shirt', 'blouse', 'top', 'sweater', 'hoodie', 'jacket', 'coat',
      'dress', 'skirt', 'jeans', 'trousers', 'shorts', 'leggings', 'kurta',
      'saree', 'jumpsuit',
    ];
    for (final c in aiCategories) {
      final m = ClothingTaxonomy.fromAiCategory(c);
      expect(m, isNotNull, reason: c);
      expect(ClothingTaxonomy.subcategories[m!.group], contains(m.sub), reason: c);
    }
    expect(ClothingTaxonomy.fromAiCategory('other'), isNull);
    expect(ClothingTaxonomy.fromAiCategory(null), isNull);
  });

  test('the picture is saved on the device and survives a restart', () async {
    final picture = base64Encode([9, 8, 7, 6]);
    final first = WardrobeStore();
    await first.add(WardrobeItem(
      id: 'p',
      filename: 'p.png',
      category: 'Top',
      subcategory: 'Shirt',
      tags: const [],
      colors: const [],
      createdAt: DateTime(2026, 1, 1),
      imageData: picture,
    ));

    final afterRestart = WardrobeStore();
    await afterRestart.load();

    expect(afterRestart.items.single.imageData, picture);
    expect(afterRestart.items.single.imageBytes, [9, 8, 7, 6]);
  });

  test('items saved by older versions (no picture data) still load', () async {
    SharedPreferences.setMockInitialValues({
      'wardrobe_items_v1':
          '[{"id":"old","filename":"old.png","category":"Top","subcategory":"Shirt"}]',
    });
    final store = WardrobeStore();
    await store.load();
    expect(store.items.single.imageData, isNull);
    expect(store.items.single.imageBytes, isNull);
  });
}
