import React, { useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  StatusBar,
  Dimensions,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';

const { width, height } = Dimensions.get('window');
const handleLogo = () => {
    navigation.navigate('GetStarted');
  }
const SplashScreen = () => {
  const navigation = useNavigation();

  useEffect(() => {
    // Auto navigate to main screen after 3 seconds
    const timer = setTimeout(() => {
      navigation.replace('FoodLens'); // Replace with your main screen name
    }, 3000);

    return () => clearTimeout(timer);
  }, [navigation]);

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#F5F5DC" />
      
      {/* Logo Container */}
      <View style={styles.logoContainer} onPress={handleLogo}>
        <Image 
          source={require('../assets/logo.png')} 
          style={styles.logo}
          resizeMode="contain"
        />
        <Text style={styles.appName}>FoodLens</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5DC', // Cream/beige background
    justifyContent: 'center',
    alignItems: 'center',
  },
  logoContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  logo: {
    width: 120,
    height: 120,
    marginBottom: 20,
  },
  appName: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#333',
    letterSpacing: 1,
  },
});

export default SplashScreen;