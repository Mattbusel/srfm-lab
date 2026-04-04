import { useContext } from 'react';
import { LiveContext } from '../context/LiveContext';

export function useLiveState() {
  const ctx = useContext(LiveContext);
  if (!ctx) throw new Error('useLiveState must be used inside LiveProvider');
  return ctx;
}
