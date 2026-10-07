import 'package:flutter/material.dart';
import 'core/constants/app_colors.dart';
import 'features/dashboard/screens/dashboard_screen.dart';
import 'features/dashboard/screens/landing_page.dart';
import 'features/wardrobe/data/wardrobe_store.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await WardrobeStore.instance.load(); // restore saved items before the first frame
  runApp(const WardrobeApp());
}

class WardrobeApp extends StatelessWidget {
  const WardrobeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Wardrobe App',
      theme: ThemeData(
        scaffoldBackgroundColor: AppColors.background,
        
        // AppBar Styling (#80110F background with White text/icons)
        appBarTheme: const AppBarTheme(
          backgroundColor: AppColors.primaryWine,
          foregroundColor: AppColors.textLight,
          elevation: 0,
          centerTitle: true,
          iconTheme: IconThemeData(color: AppColors.textLight),
          titleTextStyle: TextStyle(
            color: AppColors.textLight,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        
        // Button Theme Configuration
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: AppColors.primaryWine,
            foregroundColor: AppColors.textLight, // Text/icon on button
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),

        // Default Text Theme (Black text across app)
        textTheme: const TextTheme(
          bodyLarge: TextStyle(color: AppColors.textDark),
          bodyMedium: TextStyle(color: AppColors.textDark),
          titleLarge: TextStyle(color: AppColors.textDark, fontWeight: FontWeight.bold),
        ),
        
        useMaterial3: true,
      ),
      home: const LandingPage(),
    );
  }
}