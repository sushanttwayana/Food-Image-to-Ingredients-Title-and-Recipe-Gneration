import React, { useEffect, useRef } from 'react';
import { View, Text, Image, TouchableOpacity, StyleSheet, Dimensions, Animated } from 'react-native';
import Carousel from 'react-native-snap-carousel';
import { useNavigation } from '@react-navigation/native';

// Replace image file names with the actual filenames of the saved images
const cakeImage = require('../assets/cake.jpg');
const pastaImage = require('../assets/pasta.jpeg');
const sandwichImage = require('../assets/sandwich.jpg');

const GetStartedScreen3 = () => {
   const navigation = useNavigation();
  const sliderWidth = Dimensions.get('window').width;
  const itemWidth = 300;

  const carouselItems = [
    {
      title: 'Juicy !',
      image: sandwichImage,
    },
    {
      title: 'Fresh Pasta!',
      image: pastaImage,
    },
    {
      title: 'Sweet Cake!',
      image: cakeImage,
    },
  ];

  const ZoomImage = ({ source, style }) => {
    const scale = useRef(new Animated.Value(1)).current;

    useEffect(() => {
      const zoomInOut = Animated.loop(
        Animated.sequence([
          Animated.timing(scale, {
            toValue: 1.1, // Zoom in to 110%
            duration: 2000,
            useNativeDriver: true,
          }),
          Animated.timing(scale, {
            toValue: 1, // Zoom out back to 100%
            duration: 2000,
            useNativeDriver: true,
          }),
        ])
      );
      zoomInOut.start();
      return () => zoomInOut.stop();
    }, [scale]);

    return (
      <Animated.Image
        source={source}
        style={[style, { transform: [{ scale }] }]}
      />
    );
  };

  const renderItem = ({ item }) => (
    <View style={styles.slide}>
      <ZoomImage source={item.image} style={styles.carouselImage} />
      <Text style={styles.carouselTitle}>{item.title}</Text>
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Header Section */}
      <View style={styles.header}>
        <View style={styles.logoContainer}>
          <Image
            style={styles.logo}
            source={require('../assets/logo.png')}
          />
        </View>
        <TouchableOpacity onPress={() => navigation.navigate('Register')}>
                  <Text style={styles.skipText}>Skip</Text>
                </TouchableOpacity>
      </View>

      {/* Image Slider */}
      <Carousel
        data={carouselItems}
        renderItem={renderItem}
        sliderWidth={sliderWidth}
        itemWidth={itemWidth}
        layout={'default'}
        loop={true}
      />

      {/* Text Section */}
      <View style={styles.textContainer}>
        <Text style={styles.title}>FOOD MEETS CREATION</Text>
        <Text style={styles.greeting}>Sweet Moments!</Text>
        <View style={styles.listItem}>
          <Text style={styles.checkmark}>✔</Text>
          <Text style={styles.listText}>Upload a picture</Text>
        </View>
        <View style={styles.listItem}>
          <Text style={styles.checkmark}>✔</Text>
          <Text style={styles.listText}>Learn about the origin, ingredient</Text>
        </View>
        <View style={styles.listItem}>
          <Text style={styles.checkmark}>✔</Text>
          <Text style={styles.listText}>Recreate - Get recipes to cook at home</Text>
        </View>
        <View style={styles.listItem}>
          <Text style={styles.checkmark}>✔</Text>
          <Text style={styles.listText}>Capture your food journey</Text>
        </View>
      </View>

      {/* Next Button */}
   <TouchableOpacity 
  style={styles.button} 
  onPress={() => navigation.navigate('Register')}
>
  <Text style={styles.buttonText}>Next</Text>
</TouchableOpacity>

    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%',
    marginTop: 20,
    marginBottom: 20,
  },
  logoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  logo: {
    width: 100,
    height: 100,
    marginRight: 10,
  },
  skipText: {
    fontSize: 16,
    color: '#D32F2F',
  },
  slide: {
    alignItems: 'center',
  },
  carouselImage: {
    width: 270,
    height: 270,
    borderRadius: 10,
    marginBottom: 10,
  },
  carouselTitle: {
    fontSize: 16,
    color: '#D32F2F',
    textAlign: 'center',
  },
  textContainer: {
    alignItems: 'center',
    marginBottom: 60,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#000',
    marginBottom: 10,
  },
  greeting: {
    fontSize: 16,
    color: '#D32F2F',
    marginBottom: 10,
  },
  listItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
  },
  checkmark: {
    fontSize: 16,
    color: '#4CAF50',
    marginRight: 10,
    flexShrink: 0, // Prevents checkmark from shrinking
  },
  listText: {
    fontSize: 14,
    color: '#000',
    textAlign: 'left',
    flexShrink: 1, // Allows text to shrink if needed, but keeps it inline
    maxWidth: '80%', // Limits text width to prevent overflow
  },
  button: {
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

export default GetStartedScreen3;