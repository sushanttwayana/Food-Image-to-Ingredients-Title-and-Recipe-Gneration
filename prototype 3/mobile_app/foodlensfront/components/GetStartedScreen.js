import React, { useEffect, useRef, useState } from 'react';
import { View, Text, Image, TouchableOpacity, StyleSheet, Animated } from 'react-native';
import { useNavigation } from '@react-navigation/native';
// Replace 'food_image.png' with the actual filename of the saved image
const foodImage = require('../assets/food_image.png');

const GetStartedScreen = () => {
  const [displayText, setDisplayText] = useState('');
  const [isTyping, setIsTyping] = useState(true);
  const titleText = 'WELCOME TO FoodLens';
  const animationRef = useRef(new Animated.Value(0)).current;
  const navigation = useNavigation();


  useEffect(() => {
    const typeAndErase = () => {
      let currentIndex = 0;
      let interval;

      const type = () => {
        interval = setInterval(() => {
          if (currentIndex < titleText.length) {
            setDisplayText(titleText.substring(0, currentIndex + 1));
            currentIndex++;
          } else {
            clearInterval(interval);
            setTimeout(() => {
              setIsTyping(false);
              erase();
            }, 2000); // Pause before erasing
          }
        }, 100); // Typing speed
      };

      const erase = () => {
        interval = setInterval(() => {
          if (currentIndex > 0) {
            setDisplayText(titleText.substring(0, currentIndex - 1));
            currentIndex--;
          } else {
            clearInterval(interval);
            setTimeout(() => {
              setIsTyping(true);
              type();
            }, 500); // Pause before typing again
          }
        }, 50); // Erasing speed
      };

      type();
      return () => clearInterval(interval);
    };

    typeAndErase();
  }, []);

  // Split the displayText into parts: "WELCOME TO " and "FOODLENS"
  const welcomePart = 'WELCOME TO Food';
  const foodLensPart = 'Lens';
  const currentWelcome = displayText.substring(0, Math.min(displayText.length, welcomePart.length));
  const currentFoodLens = displayText.length > welcomePart.length 
    ? displayText.substring(welcomePart.length) 
    : '';

  return (
    <View style={styles.container}>
      {/* Logo */}
      <View style={styles.logoContainer}>
        <Image
          style={styles.logo}
          source={require('../assets/logo.png')}
        />
      </View>

      {/* Food Image */}
      <Image
        source={foodImage}
        style={styles.foodImage}
      />

      {/* Text Section */}
      <View style={styles.textContainer}>
        <Text style={styles.greeting}>Happy Foodiesss!</Text>
        <View style={styles.titleContainer}>
          <Text style={styles.titleWelcome}>{currentWelcome}</Text>
          <Text style={styles.titleFoodLens}>{currentFoodLens}</Text>
        </View>
        <Text style={styles.subtitle}>
          Where food means emotion, memories, culture, and innovation. 😋
        </Text>
        <Text style={styles.description}>
          FoodLens connects your cravings with meaning.
        </Text>
      </View>

      {/* Get Started Button */}
      {/* <TouchableOpacity style={styles.button}>
        <Text style={styles.buttonText}>Get Started</Text>
      </TouchableOpacity> */}
      <TouchableOpacity 
  style={styles.button}
  onPress={() => navigation.navigate('GetStarted1')}
>
  <Text style={styles.buttonText}>Get Started</Text>
</TouchableOpacity>

    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
    alignItems: 'center',
    padding: 20,
  },
  logoContainer: {
    marginTop: 50,
    marginBottom: 20,
  },
  logo: {
    width: 100,
    height: 100,
    marginRight: 226,
    marginTop: -20,
  },
  foodImage: {
    width: 300,
    height: 300,
    borderRadius: 10,
    marginBottom: 20,
  },
  textContainer: {
    alignItems: 'center',
    marginBottom: 30,
  },
  greeting: {
    fontSize: 16,
    color: '#D32F2F',
    marginBottom: 5,
  },
  titleContainer: {
    flexDirection: 'row',
    marginBottom: 10,
  },
  titleWelcome: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#000', // Black for "WELCOME TO "
  },
  titleFoodLens: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#D32F2F', // Red for "FOODLENS"
  },
  subtitle: {
    fontSize: 16,
    color: '#000',
    textAlign: 'center',
    marginBottom: 5,
  },
  description: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  button: {
    marginTop: 20,
    backgroundColor: '#D32F2F',
    paddingVertical: 15,
    paddingHorizontal: 60,
    borderRadius: 25,
  },
  buttonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default GetStartedScreen;