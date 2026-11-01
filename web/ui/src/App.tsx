import { createSignal, Match, onCleanup, onMount, Switch } from "solid-js";
import Demo from "./Demo";
import Writeup from "./Writeup";

function useHash() {
  const [hash, setHash] = createSignal(location.hash || "#/");
  const on = () => { setHash(location.hash || "#/"); window.scrollTo(0, 0); };
  onMount(() => window.addEventListener("hashchange", on));
  onCleanup(() => window.removeEventListener("hashchange", on));
  return hash;
}

export default function App() {
  const hash = useHash();
  const toggleTheme = () => {
    const el = document.documentElement;
    const dark = el.dataset.theme ? el.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    el.dataset.theme = dark ? "light" : "dark";
  };
  return (
    <>
      <nav class="nav">
        <span class="brand">BitNet on a Zybo</span>
        <a href="#/" classList={{ active: hash() === "#/" }}>Write-up</a>
        <a href="#/demo" classList={{ active: hash() === "#/demo" }}>Live demo</a>
        <button onClick={toggleTheme} aria-label="Toggle color theme">◐</button>
      </nav>
      <Switch fallback={<Writeup />}>
        <Match when={hash() === "#/demo"}><Demo /></Match>
      </Switch>
    </>
  );
}
