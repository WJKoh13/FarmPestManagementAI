"""Organic treatment guidance, one entry per class in detection_top15.

Written for a farmer with no technical background: what to do today, what to
use if it keeps getting worse, and when to look again. Every entry starts with
the least harmful option -- hand removal, water, barriers, traps -- and only
then names a product, because that is the order organic certification expects.

These are the offline source of truth. The local language model rephrases them
around the farmer's own note when it is running; when it is not, they render
exactly as written. Nothing here depends on the LLM being available.

Keys are the ``class_name`` slugs in data_manifests/classes_top15.json.
"""

from __future__ import annotations

GENERIC_GUIDE = """**Do today**
- Check several plants, including the undersides of leaves, and remove pests by hand where practical.
- Remove badly damaged plant parts and take them away from the field.

**Organic treatment**
- Start with the least harmful option: water spray, hand removal, barriers, or traps.
- If the problem is spreading, choose a product approved by your local organic-certification body and follow its label exactly.

**Keep watch**
- Recheck the same plants in 2 to 3 days. Take another photo if the damage increases."""


TREATMENT_GUIDES = {
    "grub": """**Do today**
- Gently dig the soil around wilting plants and hand-pick the white curled grubs you find.
- Remove dead roots and old crop waste, which is where the beetles like to lay eggs.

**Organic treatment**
- Ask a local organic supplier about beneficial nematodes suited to your pest and soil temperature. Water the soil before and after applying them, as the label directs.
- Milky spore or a Bt strain sold for grubs is an option where it is approved for your crop.

**Keep watch**
- Look again in 2 weeks. Patchy wilting that spreads in circles usually means grubs are still feeding.""",
    "mole_cricket": """**Do today**
- Look for raised tunnels in soft soil early in the morning, and for seedlings cut off near ground level.
- Pour a bucket of soapy water on a tunnel: mole crickets come to the surface within minutes and can be collected.

**Organic treatment**
- Beneficial nematodes applied to moist, warm soil in the evening work well against young crickets.
- Keep the seedbed firm and well weeded; loose, weedy soil is what they prefer.

**Keep watch**
- Check tunnels weekly. Treat while the crickets are small, before they can fly.""",
    "wireworm": """**Do today**
- Dig near weak or missing plants and look for thin, hard, yellow-brown larvae in the soil.
- Set bait traps: bury pieces of potato or carrot a few centimetres down, mark them, then lift and destroy the larvae every few days.

**Organic treatment**
- There is no quick organic spray. Control is by rotation and cultivation: avoid planting after grass or a grassy cover crop, and work the soil before planting to expose larvae to birds.

**Keep watch**
- Keep the bait traps running through the season. Rising catches mean you should not plant a susceptible crop in that field next year.""",
    "black_cutworm": """**Do today**
- Look at dusk or after dark: cutworms feed at night and hide in the top few centimetres of soil by day.
- Scrape the soil beside a cut plant, find the curled grey caterpillar, and remove it.

**Organic treatment**
- Put a stiff collar of cardboard or tin around the stem of each seedling, pushed 2 cm into the soil.
- For young caterpillars, a Bt product approved for your crop works if applied late in the day, when they come out to feed.

**Keep watch**
- Count cut plants each morning. Clear weeds early, as the moths lay eggs on them before your crop is up.""",
    "corn_borer": """**Do today**
- Check 10 plants in different parts of the field for small holes and sawdust-like droppings near stems, tassels, or cobs.
- Remove badly damaged shoots or cobs and take them away from the field.

**Organic treatment**
- For small caterpillars, apply a Bt product (Bacillus thuringiensis) approved for your crop. Aim into the leaf whorl or feeding area, preferably late afternoon.
- Treat early. It is much less effective once the caterpillars are inside the stem.

**Keep watch**
- After harvest, chop and plough in the stubble: the larvae overwinter inside old stalks.""",
    "army_worm": """**Do today**
- Check early morning or late evening, when the caterpillars feed. Look in the leaf whorl and at the field edges.
- Hand-pick where numbers are small. Army worms move in groups, so act on the whole patch at once.

**Organic treatment**
- A Bt product approved for your crop works well on small caterpillars; aim into the whorl late in the day.
- Neem-based products approved for your crop can slow young larvae.

**Keep watch**
- Numbers can rise very fast. Recheck daily during an outbreak, and warn neighbouring farms, as the moths arrive in waves.""",
    "aphids": """**Do today**
- Check the undersides of young leaves and new shoots.
- Spray small groups off with a firm stream of clean water.
- Remove only leaves that are badly covered.

**Organic treatment**
- If numbers keep rising, use insecticidal soap or neem oil labelled for your crop. Cover leaf undersides.
- Avoid spraying when bees or helpful insects are active.

**Keep watch**
- Ladybirds, lacewings and hoverfly larvae usually arrive within a week or two. If you see them, hold off spraying and let them work.""",
    "flea_beetle": """**Do today**
- Look for many small round shot-holes in young leaves, and for tiny beetles that jump when disturbed.
- Cover seedlings with a fine insect net or fleece; this is the single most effective step and needs no product.

**Organic treatment**
- Yellow sticky traps at plant height catch adults and show you how bad the pressure is.
- Where approved for your crop, a kaolin clay spray makes leaves unattractive to feed on.

**Keep watch**
- Seedlings are at risk; established plants usually outgrow the damage. Keep the net on until plants are well grown.""",
    "flax_budworm": """**Do today**
- Check buds and flower heads for small caterpillars and for holes bored into the developing bolls.
- Remove and destroy the affected heads you find.

**Organic treatment**
- Apply a Bt product approved for your crop while the caterpillars are small and still feeding on the outside of the bud.
- Pheromone traps help you time the spray for shortly after the moths appear.

**Keep watch**
- Check twice weekly through budding and flowering, which is the only window when treatment works.""",
    "tarnished_plant_bug": """**Do today**
- Check buds and growing tips for wilting, blackened spots or misshapen fruit.
- Sweep the crop with a net early in the morning while the bugs are slow, and shake them into soapy water.

**Organic treatment**
- Keep field edges mown: the bugs breed in flowering weeds and move into the crop when the weeds are cut, so mow well before the crop is vulnerable.
- Insecticidal soap or a pyrethrum product approved for your crop can help at high numbers; spray at dusk to protect bees.

**Keep watch**
- Recheck weekly during budding and early fruiting, when the damage is done.""",
    "blister_beetle": """**Do today**
- Wear gloves. These beetles release a fluid that blisters skin, and they must never be crushed onto a crop that will be fed to livestock.
- Hand-pick into soapy water, or shake plants over a bucket in the early morning.

**Organic treatment**
- A row cover keeps swarms off a vulnerable crop; the beetles arrive suddenly and in groups.
- Where approved for your crop, a pyrethrum product applied at dusk will knock down a heavy swarm.

**Keep watch**
- Their larvae eat grasshopper eggs, so a few beetles are not worth treating. Act only when a swarm is stripping foliage.""",
    "cicadella_viridis": """**Do today**
- Look on stems and leaf undersides for slender green leafhoppers that hop away when you approach.
- Check for pale stippling on the leaves and for slits in young stems where eggs are laid.

**Organic treatment**
- Yellow sticky traps reduce adults and tell you when numbers are climbing.
- Insecticidal soap or neem, approved for your crop, works on the wingless young stages. Cover leaf undersides and spray at dusk.

**Keep watch**
- Leafhoppers can carry plant diseases, so treat a rising population early rather than waiting for heavy feeding damage.""",
    "miridae": """**Do today**
- Check growing tips and flower buds for small, fast-moving bugs, and for dark sunken spots where they have fed.
- Shake plants over a light-coloured tray in the early morning to see how many there are.

**Organic treatment**
- Control flowering weeds around the field, which is where they breed and move in from.
- Insecticidal soap or a pyrethrum product approved for your crop, applied at dusk, helps at high numbers.

**Keep watch**
- Some mirids are predators that eat other pests. Check what the damage looks like before treating, and treat only where buds and tips are being lost.""",
    "prodenia_litura": """**Do today**
- Look on leaf undersides for clusters of eggs covered in brown scales, and for groups of young caterpillars feeding together.
- Remove whole leaves carrying an egg mass or a group. This is far more effective than spraying later.

**Organic treatment**
- Apply a Bt product approved for your crop against small caterpillars, late in the day.
- Pheromone traps for the adult moths show when a new generation has arrived.

**Keep watch**
- Older caterpillars feed at night and hide in the soil by day, and are much harder to control. Catch each generation while it is young.""",
    "cicadellidae": """**Do today**
- Look on leaf undersides for small wedge-shaped insects that run sideways or hop when disturbed.
- Check leaves for pale speckling, yellowing edges, or curling tips.

**Organic treatment**
- Yellow sticky traps reduce adult numbers and track the pressure.
- Insecticidal soap or neem, approved for your crop, works best on the wingless young stages. Cover leaf undersides and spray at dusk to protect bees.

**Keep watch**
- Leafhoppers spread plant diseases, so act on a rising population early. Remove crop debris and weedy margins where they shelter.""",
}


def treatment_guide(pest_name: str) -> str:
    """Guidance for a class slug, falling back to safe general advice."""
    return TREATMENT_GUIDES.get(pest_name, GENERIC_GUIDE)
