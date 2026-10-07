import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:wardrobe_app/features/dashboard/screens/dashboard_screen.dart';
import 'package:wardrobe_app/features/wardrobe/data/clothing_api.dart';
import 'package:wardrobe_app/features/wardrobe/data/wardrobe_item.dart';
import 'package:wardrobe_app/features/wardrobe/data/wardrobe_store.dart';
import 'package:wardrobe_app/features/wardrobe/screens/add_item_screen.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() async {
    SharedPreferences.setMockInitialValues({});
    await WardrobeStore.instance.load(); // resets the shared instance to empty
  });

  testWidgets('dashboard shows an empty state, then the saved item', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: DashboardScreen()));
    expect(find.text('No items yet'), findsOneWidget);

    await WardrobeStore.instance.add(WardrobeItem(
      id: '1',
      filename: 'abc.png',
      category: 'Top',
      subcategory: 'Shirt',
      tags: const ['shirt', 'navy'],
      colors: const ['navy'],
      createdAt: DateTime(2026, 1, 1),
    ));
    await tester.pump();

    expect(find.text('No items yet'), findsNothing);
    expect(find.text('Navy Shirt'), findsOneWidget);
  });

  testWidgets('Add Item: saving without a photo asks for one', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: AddItemScreen(
        api: ClothingApi(client: MockClient((_) async => http.Response('{}', 200))),
        store: WardrobeStore(),
      ),
    ));

    await tester.tap(find.byIcon(Icons.check));
    await tester.pump();

    expect(find.text('Please add a photo first.'), findsOneWidget);
  });

  testWidgets('Add Item: choosing a category swaps the subcategory choices', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: AddItemScreen(
        api: ClothingApi(client: MockClient((_) async => http.Response('{}', 200))),
        store: WardrobeStore(),
      ),
    ));
    expect(find.text('T-Shirt'), findsOneWidget);
    expect(find.text('Jeans'), findsNothing);

    await tester.tap(find.text('Bottom'));
    await tester.pump();

    expect(find.text('Jeans'), findsOneWidget);
    expect(find.text('T-Shirt'), findsNothing);
  });
}
