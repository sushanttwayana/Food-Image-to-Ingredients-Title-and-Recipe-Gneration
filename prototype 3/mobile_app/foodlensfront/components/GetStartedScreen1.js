import React from 'react';
import { View, Text, Image, TouchableOpacity, StyleSheet, Dimensions } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Carousel from 'react-native-snap-carousel';

// Replace image file names with the actual filenames of the saved images
const burgerImage = require('../assets/burger.png');
const yomariImage = require('../assets/yomari.png');
const pizzaImage = require('../assets/pizza.jpg');


const GetStartedScreen1 = () => {
  const navigation = useNavigation();
  const sliderWidth = Dimensions.get('window').width;
  const itemWidth = 300;
  

  const carouselItems = [
    {
      title: 'Juicy Burger!',
      image: burgerImage,
    },
    {
      title: 'Fresh Yomari!',
      image: yomariImage,
    },
    {
      title: 'Tasty Pizza!',
      image: pizzaImage,
    },
  ];

  const renderItem = ({ item }) => (
    <View style={styles.slide}>
      <Image source={item.image} style={styles.carouselImage} />
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
         <TouchableOpacity onPress={() => navigation.navigate('GetStarted3')}>
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
        <Text style={styles.title}>FOOD MEETS MEMORY</Text>
        <Text style={styles.subtitle}>
          Discover the taste of tradition and the power of technology—right at your fingertips.
        </Text>
        <Text style={styles.description}>
          Because at FoodLens, every dish tells a story.
        </Text>
      </View>

      {/* Next Button */}
      {/* <TouchableOpacity style={styles.button}>
        <Text style={styles.buttonText}>Next</Text>
        
      </TouchableOpacity> */}
        <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('GetStarted3')}
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
    padding: 40,
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
    width: 300,
    height: 300,
    borderRadius: 10,
    marginBottom: 10,
  },
  carouselTitle: {
    fontSize: 20,
    color: '#D32F2F',
    textAlign: 'center',
  },
  textContainer: {
    alignItems: 'center',
   marginBottom: 65,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#000',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 14,
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

export default  GetStartedScreen1 ;